import scipy.optimize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---
from utils.graph_utils import (
    adj_to_edge_index,
    triu_to_3d_dense,
    recover_adj_lower,
    recover_full_adj_from_lower,
)


class AdaptiveSigmoid(nn.Module):
    def __init__(self, lambda_init=1.0):
        super(AdaptiveSigmoid, self).__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lambda_param * self.sigmoid(x)
        return x


class AdaptiveRelu(nn.Module):
    def __init__(self, lambda_init=1.0):
        super(AdaptiveSigmoid, self).__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lambda_param * self.relu(x)
        return x


# Soluzione 2: Weighted loss
class WeightedBCELoss(nn.Module):

    def __init__(self, weight_positive=2.0):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive

    def forward(self, pred, target):
        loss = F.binary_cross_entropy(pred, target, reduction="none")
        weight = torch.ones_like(target)
        weight[target > 0] = self.weight_positive
        weighted_loss = loss * weight
        return weighted_loss.mean()


class GraphVAE_loss(nn.Module):

    def __init__(
        self,
        recon_loss_fn,
        recon_w: float = 1.0,
        node_w: float = 1.0,
        edge_w: float = 1.0,
        adj_w: float = 1.0,
        kl_w: float = 1.0,
    ):
        super(GraphVAE_loss, self).__init__()
        self.recon_w = recon_w
        self.node_w = node_w
        self.edge_w = edge_w
        self.adj_w = adj_w
        self.kl_w = kl_w
        self.recon_loss_fn = recon_loss_fn
        self.adj_loss_fn = WeightedBCELoss(weight_positive=2.0)
        # self.adaptive_kl_w = nn.Parameter(torch.tensor(1.0))

    def calc_loss(
        self,
        adj_true,
        adj_recon,
        node_true,
        node_recon,
        edges_true,
        edges_recon,
        batch_index,
        mu,
        var,
        fingerprint_model=None,
        normalize=False,
    ):

        # reshape dell'output della VAE in modo da ottenere risultati in forma matriciale
        self.max_num_nodes = node_recon.size(1)

        node_recon = node_recon.view(node_true.shape)
        edges_recon = edges_recon.view(edges_true.shape)

        adj_recon_lower = recover_adj_lower(adj_recon, self.max_num_nodes)
        adj_recon_full = recover_full_adj_from_lower(adj_recon_lower)
        adj_true = adj_true.view(adj_recon_full.shape)  # reshape in forma matriciale 9x9

        self.loss_edge = F.mse_loss(edges_recon, edges_true)
        # self.loss_edge = self.adj_loss_fn(edges_recon, edges_true)
        self.loss_node = F.mse_loss(node_recon, node_true)
        # print("-" * 20)
        # print(adj_true[0])
        # print("*" * 20)
        # print(adj_recon_full[0])

        # self.loss_adj = F.mse_loss(adj_recon_full, adj_true)
        # self.loss_adj = self.adj_loss_fn(adj_recon_full, adj_true)
        self.loss_adj = adj_vec_loss(adj_recon_full, adj_true)

        self.loss_kl = kl_loss(mu, var)

        loss_name = self.recon_loss_fn.__name__
        if "fingerprint" in loss_name:
            self.loss_recon = self.recon_loss_fn(
                fingerprint_model, adj_true, adj_recon_full, node_true, node_recon, batch_index
            )
        else:
            self.loss_recon = self.recon_loss_fn(
                adj_true,
                adj_recon_full,
                node_true,
                node_recon,
                edges_true,
                edges_recon,
                batch_index,
                max_num_nodes=self.max_num_nodes,
            )

        if normalize:
            loss_mean = torch.mean(
                torch.tensor([self.loss_edge, self.loss_node, self.loss_adj, self.loss_kl, self.loss_recon])
            )
            loss_std = torch.std(
                torch.tensor([self.loss_edge, self.loss_node, self.loss_adj, self.loss_kl, self.loss_recon])
            )
            self.loss_edge = (self.loss_edge - loss_mean) / loss_std
            self.loss_node = (self.loss_node - loss_mean) / loss_std
            self.loss_adj = (self.loss_adj - loss_mean) / loss_std
            self.loss_kl = (self.loss_kl - loss_mean) / loss_std
            self.loss_recon = (self.loss_recon - loss_mean) / loss_std

        self.loss_edge = self.loss_edge * self.edge_w
        self.loss_node = self.loss_node * self.node_w
        self.loss_adj = self.loss_adj * self.adj_w
        self.loss_kl = self.loss_kl * self.kl_w
        self.loss_recon = self.loss_recon * self.recon_w

        self.total_loss = self.loss_edge + self.loss_node + self.loss_adj + self.loss_kl + self.loss_recon

        self.dict_losses = {
            "total_loss": self.total_loss,
            "loss_edge": self.loss_edge,
            "loss_node": self.loss_node,
            "loss_adj": self.loss_adj,
            "loss_kl": self.loss_kl,
            "loss_recon": self.loss_recon,
        }

        return self.total_loss, self.dict_losses


def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    # loss_kl = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    # loss_kl /= self.max_num_nodes * self.max_num_nodes
    # x_sigma = torch.exp(logstd)
    # kl_div = (x_sigma**2 + mu**2 - torch.log(x_sigma) - 0.5).sum()
    kl_div = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())

    # MAX_LOGSTD = 10
    # logstd = logstd.clamp(max=MAX_LOGSTD)
    # kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp() ** 2, dim=1))

    # # Limit numeric errors
    # kl_div = kl_div.clamp(max=1000)
    return kl_div


def adj_myversion_loss(A_pred, A_real):
    # Dimensioni
    batch_size = A_pred.shape[0]
    k = A_pred.shape[1]

    # Loss per la matrice di adiacenza
    loss_A = 0
    off_diag_term = F.binary_cross_entropy(A_pred, A_real, reduction="sum")
    loss_A += off_diag_term / (batch_size * k * (k - 1))

    return loss_A


def adj_vec_loss(A_pred, A_real):
    batch_size, n, _ = A_pred.shape

    # Creiamo una maschera per la triangolare superiore (escludendo la diagonale)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool().unsqueeze(0).expand(batch_size, -1, -1)

    # Applichiamo la maschera e convertiamo in vettori
    pred_flat = A_pred[mask].view(batch_size, -1)
    target_flat = A_real[mask].view(batch_size, -1)

    # Calcoliamo la binary cross entropy
    bce_loss = F.binary_cross_entropy(pred_flat, target_flat, reduction="mean")

    return bce_loss


def approx_loss(
    adj_true, adj_recon, node_true, node_recon, edges_true, edges_recon, batch_index, max_num_nodes
):

    # Reconstruction loss per graph
    batch_recon_loss = []

    batch_size = len(torch.unique(batch_index))
    node_true = node_true.view([adj_true.shape[0], -1, node_true.shape[-1]])
    edges_true = edges_true.view([adj_true.shape[0], -1, edges_true.shape[-1]])
    node_recon = node_recon.view([adj_recon.shape[0], -1, node_recon.shape[-1]])
    edges_recon = edges_recon.view([adj_recon.shape[0], -1, edges_recon.shape[-1]])

    output_node_recon_features = F.softmax(node_recon[:, :, 5:9], dim=2)
    output_node_true_features = node_true[:, :, 5:9]

    for i in torch.unique(batch_index):  # per ogni batch

        # Calculate losses
        recon_loss = approximate_recon_loss(
            output_node_true_features[i], output_node_recon_features[i], edges_true[i], edges_recon[i], max_num_nodes=max_num_nodes
        )
        batch_recon_loss.append(recon_loss)

    # Take average of all losses
    batch_recon_loss = torch.true_divide(sum(batch_recon_loss), batch_size)

    return batch_recon_loss


def fingerprint_loss(model_fingerprint, adj_true, adj_recon, node_true, node_recon, batch_index):

    # prima passo da adj a edge_index
    edge_index_recon, _ = adj_to_edge_index(adj_recon)
    edge_index_true, _ = adj_to_edge_index(adj_true)

    batch_index = torch.tensor([len(batch_index)], device=adj_true.device)

    fingerprint_true = model_fingerprint(node_true, edge_index_true, batch_index)
    fingerprint_recon = model_fingerprint(node_recon, edge_index_recon, batch_index)

    fingerprint_recon_loss = F.mse_loss(fingerprint_recon, fingerprint_true)

    return fingerprint_recon_loss


def matching_loss(
    adj_true, adj_recon_vector, node_true, node_recon, edges_true, edges_recon, batch_index, max_num_nodes
):

    node_true = node_true.view(node_recon.shape)
    edges_true = edges_true.view(edges_recon.shape)
    adj_true = adj_true.view(-1, max_num_nodes, max_num_nodes)

    edges_recon_features_total = torch.empty(
        adj_true.shape[0],
        edges_recon.shape[1],
        edges_recon.shape[2],
        device=adj_true.device,
    )

    upper_triangular_indices = torch.triu_indices(
        row=adj_true[0].size(0),
        col=adj_true[0].size(1),
        offset=1,
        device=adj_true.device,
    )

    init_corr = 1 / max_num_nodes

    adj_permuted_vectorized = adj_recon_vector.clone().to(adj_true.device)
    recon_adj_lower = recover_adj_lower(adj_recon_vector)
    recon_adj_tensor = recover_full_adj_from_lower(recon_adj_lower)

    # LENTISSIMOO...
    for i in range(adj_recon_vector.shape[0]):  # per ogni batch

        adj_wout_diagonal = adj_true[i][
            upper_triangular_indices[0], upper_triangular_indices[1]
        ]  # rimuovo diagonale
        adj_mask = adj_wout_diagonal.repeat(edges_recon.shape[2], 1).T  # ripeto per le righe
        masked_edges_recon_features = edges_recon[i] * adj_mask  # rimuovo le righe con diagonale 0
        edges_recon_features_total[i] = masked_edges_recon_features.reshape(
            -1, edges_recon.shape[2]
        )  # reshape

        S = edge_similarity_matrix(
            adj_true[i],
            recon_adj_tensor[i],
            edges_true[i],
            edges_recon[i],
            deg_feature_similarity,
        )

        init_assignment = torch.ones(max_num_nodes, max_num_nodes, device=adj_true.device) * init_corr
        assignment = mpm(init_assignment, S)

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.detach().cpu().numpy())
        # Algoritmo ungherese implementato in torch per velocizzare le operazioni e fare tutto su gpu
        # row_ind, col_ind = hungarian_algorithm(assignment)

        adj_permuted = permute_adj(adj_true[i], row_ind, col_ind)
        adj_permuted_vectorized[i] = adj_permuted[torch.triu(torch.ones(max_num_nodes, max_num_nodes)) == 1]

    adj_recon_loss = F.binary_cross_entropy(adj_recon_vector, adj_permuted_vectorized)

    return adj_recon_loss


def approximate_recon_loss(node_targets, node_preds, triu_targets, triu_preds, max_num_nodes):
    """
    See: https://github.com/seokhokang/graphvae_approx/
    TODO: Improve loss function
    """

    # Apply sum on labels per (node/edge) type and discard "none" types
    node_preds_reduced = torch.sum(node_preds[:, :], 0)
    node_targets_reduced = torch.sum(node_targets, 0)
    triu_preds_reduced = torch.sum(triu_preds[:, :], 0)
    triu_targets_reduced = torch.sum(triu_targets, 0)

    # Calculate node-sum loss and edge-sum loss
    node_loss = sum(squared_difference(node_preds_reduced, node_targets_reduced.float()))
    edge_loss = sum(squared_difference(triu_preds_reduced, triu_targets_reduced.float()))

    # Calculate node-edge-sum loss
    # Forces the model to properly arrange the matrices
    node_edge_loss = calculate_node_edge_pair_loss(
        node_targets, triu_targets, node_preds, triu_preds, max_num_nodes
    )

    approx_loss = node_edge_loss + edge_loss + node_loss
    if all(node_targets_reduced == node_preds_reduced.int()) and all(
        triu_targets_reduced == triu_preds_reduced.int()
    ):
        print("Reconstructed all edges: ", node_targets_reduced)
        print("and all nodes: ", node_targets_reduced)
    return approx_loss


def calculate_node_edge_pair_loss(node_tar, edge_tar, node_pred, edge_pred, max_num_nodes):

    SUPPORTED_EDGES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    node_features = node_pred.shape[1]

    # Recover full 3d adjacency matrix for edge predictions
    edge_pred_3d = triu_to_3d_dense(
        edge_pred, node_pred.shape[0], device=node_pred.device
    )  # [num nodes, num nodes, edge types]

    # Recover full 3d adjacency matrix for edge targets
    edge_tar_3d = triu_to_3d_dense(
        edge_tar, node_tar.shape[0], device=node_pred.device
    )  # [num nodes, num nodes, edge types]

    # --- The two output matrices tell us how many edges are connected with each of the atom types
    # Multiply each of the edge types with the atom types for the predictions
    node_edge_preds = torch.empty(
        [max_num_nodes, node_features, len(SUPPORTED_EDGES)], dtype=torch.float, device=node_pred.device
    )

    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_preds[:, :, edge] = torch.matmul(edge_pred_3d[:, :, edge], node_pred[:, :])

    # Multiply each of the edge types with the atom types for the targets
    node_edge_tar = torch.empty(
        (node_tar.shape[0], node_features, len(SUPPORTED_EDGES)), dtype=torch.float, device=node_pred.device
    )
    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_tar[:, :, edge] = torch.matmul(edge_tar_3d[:, :, edge], node_tar.float())

    # Reduce to matrix with [num atom types, num edge types]
    node_edge_pred_matrix = torch.sum(node_edge_preds, dim=0)
    node_edge_tar_matrix = torch.sum(node_edge_tar, dim=0)

    if torch.equal(node_edge_pred_matrix.int(), node_edge_tar_matrix.int()):
        print("Reconstructed node-edge pairs: ", node_edge_pred_matrix.int())

    node_edge_loss = torch.mean(sum(squared_difference(node_edge_pred_matrix, node_edge_tar_matrix.float())))

    # Calculate node-edge-node for preds
    node_edge_node_preds = torch.empty(
        (max_num_nodes, max_num_nodes, len(SUPPORTED_EDGES)), dtype=torch.float, device=node_pred.device
    )
    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_node_preds[:, :, edge] = torch.matmul(node_edge_preds[:, :, edge], node_pred[:, :].t())

    # Calculate node-edge-node for targets
    node_edge_node_tar = torch.empty(
        (node_tar.shape[0], node_tar.shape[0], len(SUPPORTED_EDGES)),
        dtype=torch.float,
        device=node_pred.device,
    )
    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_node_tar[:, :, edge] = torch.matmul(node_edge_tar[:, :, edge], node_tar.float().t())

    # Node edge node loss
    node_edge_node_loss = sum(
        squared_difference(torch.sum(node_edge_node_preds, [0, 1]), torch.sum(node_edge_node_tar, [0, 1]))
    )

    # TODO: Improve loss
    return node_edge_loss  #  * node_edge_node_loss


def permute_adj(adj, curr_ind, target_ind, max_num_nodes):
    """Permute adjacency matrix.
    The target_ind (connectivity) should be permuted to the curr_ind position.
    """
    # order curr_ind according to target ind
    ind = np.zeros(max_num_nodes, dtype=np.int32)
    ind[target_ind] = curr_ind
    adj_permuted = torch.zeros((max_num_nodes, max_num_nodes), device=adj.device)
    adj_permuted[:, :] = adj[ind, :]
    adj_permuted[:, :] = adj_permuted[:, ind]
    return adj_permuted


def deg_feature_similarity(f1, f2):
    edge_similarity = F.cosine_similarity(f1, f2, dim=0)
    return edge_similarity


def edge_similarity_matrix(
    adj, adj_recon, matching_features, matching_features_recon, sim_func, max_num_nodes
):
    S = torch.zeros(
        max_num_nodes,
        max_num_nodes,
        max_num_nodes,
        max_num_nodes,
        device=adj.device,
    )

    for i in range(max_num_nodes):
        for j in range(max_num_nodes):
            if i == j:

                for a in range(max_num_nodes):

                    # calcolo la similarità nei loop
                    try:
                        S[i, i, a, a] = (
                            adj[i, i]
                            * adj_recon[a, a]
                            * sim_func(matching_features[i], matching_features_recon[a])
                        )
                    except:
                        S[i, i, a, a] = 0

            else:
                for a in range(max_num_nodes):
                    for b in range(max_num_nodes):
                        if b == a:
                            continue
                        S[i, j, a, b] = torch.abs(adj[i, j] - adj_recon[a, b])
    return S


def mpm(x_init, S, max_iters=10, max_num_nodes=9):
    x = x_init
    for it in range(max_iters):
        x_new = torch.zeros(max_num_nodes, max_num_nodes, device=x.device)
        for i in range(max_num_nodes):
            for a in range(max_num_nodes):
                x_new[i, a] = x[i, a] * S[i, i, a, a]
                pooled = [torch.max(x[j, :] * S[i, j, a, :]) for j in range(max_num_nodes) if j != i]
                neigh_sim = sum(pooled)
                x_new[i, a] += neigh_sim

        norm = torch.norm(x_new)

        x = x_new / norm

    return x


def squared_difference(input, target):
    return (input - target) ** 2

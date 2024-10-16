import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import os

# ---
from GraphVAE.encoder import VAE_conv_ENCODER_2, VAE_conv_ENCODER, VAE_conv_ENCODER_3
from GraphVAE.decoder import VAE_plain_DECODER, VAE_plain_DECODER_2
from utils.graph_utils import recover_adj_lower, recover_full_adj_from_lower

a = sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class GraphVAE(nn.Module):
    def __init__(
        self,
        latent_dim,
        max_num_nodes,
        max_num_edges,
        num_nodes_features=11,
        num_edges_features=4,
        num_layers_encoder=5,
        num_layers_decoder=3,
        growth_factor_encoder=2,
        growth_factor_decoder=1,
        pool="sum",
    ):

        super(GraphVAE, self).__init__()

        self.num_nodes_features = num_nodes_features
        self.num_edges_features = num_edges_features

        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.latent_dim = latent_dim

        # self.h_size = (
        #     self.max_num_nodes * self.num_nodes_features
        # )  # dimensione dell'input, in questo caso schiaccio la matrice delle features

        # dimensione dell'input, in questo caso sono il numero di featrues per nodo
        self.h_size = self.num_nodes_features

        # dimensione dell'output della matrice delle features dei nodi
        self.output_node_features = self.max_num_nodes * self.num_nodes_features
        self.embedding_size = latent_dim
        # dimensione dell'output della matrice delle features degli edges
        self.e_size = self.max_num_edges * self.num_edges_features
        self.output_adj_vec_dim = max_num_nodes * (max_num_nodes + 1) // 2

        # definizione dei componenti della VAE:
        # self.encoder = VAE_plain_ENCODER(self.h_size, self.embedding_size, device).to(device)
        self.encoder = VAE_conv_ENCODER_2(
            self.h_size,
            self.embedding_size,
            num_edges_features,
            num_layers=num_layers_encoder,
            dropout=0.001,
            growth_factor=growth_factor_encoder,
        )

        self.decoder = VAE_plain_DECODER_2(
            self.embedding_size,
            self.output_adj_vec_dim,
            self.output_node_features,
            self.e_size,
            num_layers=num_layers_decoder,
            dropout=0.001,
            growth_factor=growth_factor_decoder,
        )

        self.pool = pool

    def pool_graph(self, x):
        if self.pool == "max":
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == "sum":
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, nodes_features, edge_attr, edge_index, batch_index):
        # graph_h = nodes_features.reshape(
        #     -1, self.max_num_nodes * self.num_nodes_features
        # )  # spiano la matrice delle features dei nodi

        nodes_features = nodes_features.view(-1, self.num_nodes_features)

        # VAE:
        z, z_mu, z_lsgms = self.encoder(nodes_features, edge_attr, edge_index, batch_index)
        h_decode, node_recon_features, edges_recon_features = self.decoder(z)

        # reshape dell'output della VAE in modo da ottenere risultati in forma matriciale
        node_recon_features = node_recon_features.view(-1, self.max_num_nodes, self.num_nodes_features)

        edges_recon_features = edges_recon_features.view(-1, self.max_num_edges, self.num_edges_features)

        # softmax in modo da avere valori probabilistici per la matrice delle features degli edges
        edges_recon_features = F.softmax(edges_recon_features, dim=2)

        # faccio la sigmoid perchè i valori dell'adiacente sono compresi tra 0 e 1
        out = F.sigmoid(h_decode)

        return out, z_mu, z_lsgms, node_recon_features, edges_recon_features

    def generate(self, z, treshold_adj=0.50, treshold_diag=0.50):

        with torch.no_grad():
            h_decode, output_node_features, output_edge_features = self.decoder(z)
            output_node_features = output_node_features.view(-1, self.max_num_nodes, self.num_nodes_features)
            output_edge_features = output_edge_features.view(-1, self.max_num_edges, self.num_edges_features)

            # degli edge faccio la softmax perchè erano codificati in one-hot
            output_edge_features = F.softmax(output_edge_features, dim=2)

            # della matrice di adiacenza faccio la sigmoid per schiacciare i valori tra 0 e 1
            out = torch.sigmoid(h_decode)

            # ricostruisco la matrice di adiacenza:
            recon_adj_lower = recover_adj_lower(out, self.max_num_nodes)
            recon_adj_tensor = recover_full_adj_from_lower(recon_adj_lower)
            # Trova gli indici delle righe e delle colonne con valori diagonali superiori a treshold_diag
            diagonal_values = torch.diagonal(recon_adj_tensor, dim1=-2, dim2=-1)
            indices_bool = diagonal_values > treshold_diag
            # Crea la maschera booleana per selezionare le righe e le colonne
            mask = indices_bool.unsqueeze(1) & indices_bool.unsqueeze(2)
            # Seleziona le righe e le colonne
            selected_matrices = recon_adj_tensor * mask
            # Ottenere le dimensioni del tensore
            _, matrix_size, _ = selected_matrices.size()
            # Creare una maschera identità per le diagonali
            diagonal_mask = torch.eye(matrix_size, device=z.device).bool()
            # Moltiplicare ogni matrice per la maschera identità invertita
            result_tensor = selected_matrices * (~diagonal_mask).unsqueeze(0)
            # arrotondo la matrice adiacente ad una treshold definita
            recon_adj_tensor_rounded = torch.round(result_tensor + (0.5 - treshold_adj))

            # conto il numeri di 1 per ogni matrice
            n_one = torch.sum(recon_adj_tensor_rounded, dim=(1, 2)) // 2

            sotto_matrice_nodi = F.softmax(output_node_features[:, :, 5:9], dim=2)
            output_node_features[:, :, 5:9] = sotto_matrice_nodi.squeeze_()

            return (
                recon_adj_tensor_rounded,
                output_node_features,
                output_edge_features,
                n_one,
            )

    def freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


def load_graphvae_from_folder(model_folder: str = ""):
    """
    A function to load a GraphVAE model from the specified model folder.

    Parameters:
    - model_folder (str): The path to the folder containing the model files.

    Returns:
    - model_GraphVAE (GraphVAE): The loaded GraphVAE model.
    - encoder (MLP_VAE_plain_ENCODER): The encoder model associated with the GraphVAE.
    - hyper_params (dict): The hyperparameters used for the model.
    """

    if not os.path.exists(model_folder):
        print("VAE non trovato...")
        exit()

    json_files = "hyperparams.json"
    # Cerca i file con estensione .pth
    pth_files = [file for file in os.listdir(model_folder) if file.endswith("FINAL.pth")]

    if json_files:
        hyper_file_json = os.path.join(model_folder, json_files)
    else:
        hyper_file_json = None

    if pth_files:
        model_file_pth = os.path.join(model_folder, pth_files[0])
    else:
        model_file_pth = None

    with open(hyper_file_json, "r") as file:
        hyper_params = json.load(file)[0]

    model_GraphVAE = GraphVAE(
        latent_dim=hyper_params["latent_dimension"],
        max_num_nodes=hyper_params["max_num_nodes"],
        max_num_edges=hyper_params["max_num_edges"],
        num_nodes_features=hyper_params["num_nodes_features"],
        num_edges_features=hyper_params["num_edges_features"],
        num_layers_decoder=hyper_params["num_layers_decoder"],
        num_layers_encoder=hyper_params["num_layers_encoder"],
        growth_factor_decoder=hyper_params["growth_factor_decoder"],
        growth_factor_encoder=hyper_params["growth_factor_encoder"],
    )

    data_model = torch.load(model_file_pth, map_location="cpu", weights_only=True)
    model_GraphVAE.load_state_dict(data_model)

    return model_GraphVAE, hyper_params

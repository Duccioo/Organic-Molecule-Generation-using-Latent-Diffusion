import os
import numpy as np
import networkx as nx
import pickle

import torch
from torch_geometric.utils import to_dense_adj

import rdkit.Chem as Chem
from rdkit import rdBase

from tqdm import tqdm

# ---
from utils.config import MAX_MOLECULE_SIZE, SUPPORTED_EDGES


blocker = rdBase.BlockLogs()


def count_edges(adj_matrix):
    num_edges = torch.sum(adj_matrix) / 2  # Diviso per 2 perché la matrice adiacente è simmetrica
    return num_edges


def matrix_graph_to_mol(adj, node_labels, edge_features, sanitize=False, cleanup=False):
    """
    Converts a graph into an RDKit molecule.

    Parameters
    ----------
    adj : torch.tensor
        Adjacency matrix of the graph.
    node_labels : torch.tensor
        Node labels of the graph.
    edge_features : torch.tensor
        Edge features of the graph.
    sanitize : bool
        If True, sanitize the molecule with RDKit.
    cleanup : bool
        If True, remove all properties from the molecule before returning it.

    Returns
    -------
    G : networkx.Graph
        The input graph.
    mol : RDKit Chem.Mol
        The RDKit molecule object.
    """

    G = matrix_graph_to_nx(adj, node_labels, edge_features, filename=None)

    mol = nx_to_mol(G)

    return G, mol


def nx_to_mol(nx_graph, atomic_number: dict = None, bond_type: dict = None, atom_label: dict = None):
    DEFAULT_atomic_number = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    DEFAULT_atom_label = {0: "C", 1: "N", 2: "O", 3: "F"}
    DEFAULT_bond_type = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    if atomic_number is None:
        atomic_number = DEFAULT_atomic_number
    if bond_type is None:
        bond_type = DEFAULT_bond_type

    if atom_label is None:
        atom_label = DEFAULT_atom_label

    mol = Chem.RWMol()
    # define two dicts to keep track of the atom index corresponding to each node index
    atom_to_node = dict()
    node_to_atom = dict()
    # create an atom object for each node in the graph

    for j, node in enumerate(nx_graph.nodes()):
        node_features = nx_graph.nodes[node]["features"]

        a = Chem.Atom(atomic_number[atom_label[node_features]])
        a.SetFormalCharge(0)
        index = mol.AddAtom(a)

        atom_to_node[index] = j
        node_to_atom[node] = index

    # copy the bonds
    for e, edge in enumerate(nx_graph.edges()):

        atom0 = node_to_atom[edge[0]]
        atom1 = node_to_atom[edge[1]]

        num_label = nx_graph.edges[edge[0], edge[1]]["features"] + 1
        if isinstance(num_label, np.ndarray):
            num_label = num_label[0]
        mol.AddBond(atom0, atom1, bond_type[num_label])

    # append the new molecule object to the list
    return mol


def graph_to_mol_old(adj, node_labels, edge_features, sanitize, cleanup):
    mol = Chem.RWMol()
    smiles = ""

    node_labels = node_labels[:, 5:9]
    node_labels = torch.argmax(node_labels, dim=1)

    atomic_numbers = {0: "C", 1: "N", 2: "O", 3: "F"}

    # Crea un dizionario per mappare la rappresentazione one-hot encoding ai tipi di legami
    bond_types = {
        0: Chem.rdchem.BondType.SINGLE,
        1: Chem.rdchem.BondType.DOUBLE,
        2: Chem.rdchem.BondType.TRIPLE,
        3: Chem.rdchem.BondType.AROMATIC,
    }
    # print(f"Node Labels {node_labels}")

    idx = 0

    for node_label in node_labels.tolist():
        # print(f"Adding atom {atomic_numbers[node_label]}")
        mol.AddAtom(Chem.Atom(atomic_numbers[node_label]))

    for edge in np.nonzero(adj).tolist():
        start, end = edge[0], edge[1]
        if start > end:

            bond_type_one_hot = int((edge_features[idx]).argmax())
            bond_type = bond_types[bond_type_one_hot]
            # print(f"ADDING BOUND {bond_type} to {start} and {end}")
            idx += 1

            try:
                mol.AddBond(int(start), int(end), bond_type)
            except:
                print("ERROR Impossibile aggiungere legame")
    if sanitize:
        try:
            flag = Chem.SanitizeMol(mol, catchErrors=True)
            # Let's be strict. If sanitization fails, return None
            if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                mol = None

                # print("Sanitize Failed")

        except Exception:
            # print("Sanitize Failed")
            mol = None

    if cleanup and mol is not None:
        try:
            mol = Chem.AddHs(mol, explicitOnly=True)
        except:
            pass

        try:
            smiles = Chem.MolToSmiles(mol)
            smiles = max(smiles.split("."), key=len)
            if "*" not in smiles:
                mol = Chem.MolFromSmiles(smiles)
            else:
                print("mol from smiles failed")
                mol = None
        except:
            # print("error generic")
            smiles = Chem.MolToSmiles(mol)

            mol = None

    return mol, smiles


def slice_graph_targets(graph_id, edge_targets, node_targets, batch_index):
    """
    Slices out the upper triangular part of an adjacency matrix for
    a single graph from a large adjacency matrix for a full batch.
    For the node features the corresponding section in the batch is sliced out.
    --------
    graph_id: The ID of the graph (in the batch index) to slice
    edge_targets: A dense adjacency matrix for the whole batch
    node_targets: A tensor of node labels for the whole batch
    batch_index: The node to graph map for the batch
    """
    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    # Row slice and column slice batch targets to get graph edge targets
    graph_edge_targets = edge_targets[graph_mask][:, graph_mask]
    # Get triangular upper part of adjacency matrix for targets
    size = graph_edge_targets.shape[0]
    triu_indices = torch.triu_indices(size, size, offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    graph_edge_targets = graph_edge_targets[triu_mask]
    # Slice node targets
    graph_node_targets = node_targets[graph_mask]
    return graph_edge_targets, graph_node_targets


def slice_graph_predictions(
    triu_logits, node_logits, graph_triu_size, triu_start_point, graph_size, node_start_point
):
    """
    Slices out the corresponding section from a list of batch triu values.
    Given a start point and the size of a graph's triu, simply slices
    the section from the batch list.
    -------
    triu_logits: A batch of triu predictions of different graphs
    node_logits: A batch of node predictions with fixed size MAX_GRAPH_SIZE
    graph_triu_size: Size of the triu of the graph to slice
    triu_start_point: Index of the first node of this graph in the triu batch
    graph_size: Max graph size
    node_start_point: Index of the first node of this graph in the nodes batch
    """
    # Slice edge logits
    graph_logits_triu = torch.squeeze(triu_logits[triu_start_point : triu_start_point + graph_triu_size])
    # Slice node logits
    graph_node_logits = torch.squeeze(node_logits[node_start_point : node_start_point + graph_size])
    return graph_logits_triu, graph_node_logits


def matrix_graph_to_nx(adj_matrix, node_features, edge_features, filename=None):

    # sposto le features in una lista e le converto in numpy per evitare problemi con i tensori in PyTorch
    # prima arrotondo la matrice adiacente
    # adj_matrix = torch.round(adj_matrix)
    adj_matrix_np = adj_matrix.detach().cpu().numpy()
    if adj_matrix_np.shape[0] == 1:
        adj_matrix_np = np.squeeze(adj_matrix_np, axis=0)

    # Per le features dei nodi e degli edges faccio anche un argmax e le converto in numpy
    if node_features.shape[0] == 1:
        node_features = node_features.squeeze(0)

    node_features = node_features[:, 5:9]
    node_features = torch.argmax(node_features, dim=1)
    node_features_np = node_features.detach().cpu().numpy()

    if edge_features.shape[0] == 1:
        edge_features = edge_features.squeeze(0)

    edge_features = torch.argmax(edge_features, dim=1)
    edge_features_np = edge_features.detach().cpu().numpy()

    G = nx.from_numpy_array(adj_matrix_np)

    # Aggiunta delle features dei nodi
    for i, features in enumerate(node_features_np):
        G.nodes[i]["features"] = features

    # # Aggiunta delle features degli edge
    # for i, j in zip(*adj_matrix_np.nonzero()):
    #     if i > j:

    #         G.edges[i, j]["features"] = edge_features_np[i]

    # Aggiungere archi con le loro features
    num_nodes = adj_matrix_np.shape[0]
    edge_index = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Evita duplicati e self-loop
            if adj_matrix_np[i, j] != 0:
                # G.add_edge(i, j, edge_features_np[edge_index])
                G.edges[i, j]["features"] = edge_features_np[edge_index]
                edge_index += 1

    nodes_without_edges = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(nodes_without_edges)

    # check if the graph can be a molecule
    # try:
    #     mol = nx_to_mol(G)
    # except:
    #     print("Attributi degli archi:")
    #     for u, v, data in G.edges(data=True):
    #         print(f"Arco ({u}, {v}): {data}")

    #     print(edge_features_np)
    #     print("ADJ", adj_matrix_np)
    #     print("NODES", node_features_np)
    #     print("Graph cannot be a molecule, skipping")
    #     exit()

    # Salva il grafo in formato .pkl
    # nx.write_gpickle(G, filename)
    if filename is not None:
        with open(filename, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    return G


def save_graphs_to_file(list_graphs: list[nx.Graph], filename: str = "graphs_generated.pkl"):

    list_of_dicts = [nx.to_dict_of_dicts(graph) for graph in list_graphs]

    # Salva i grafi nel file
    with open(filename, "wb") as file:
        pickle.dump(list_of_dicts, file, pickle.HIGHEST_PROTOCOL)

    print(f"Grafi salvati nel file '{filename}'.")


def load_graphs_from_folder(folder_path: str = "graphs_generated", create_using=nx.Graph):

    graph_list = []
    graph_paths = os.listdir(folder_path)
    for i, g in tqdm(enumerate(graph_paths), total=len(graph_paths), desc="Loading graphs"):
        # print("Loading graph " + str(i + 1) + " of " + str(len(folder_path)), end="\r")
        graph_path = os.path.join(folder_path, f"G_{i}", "graph.pkl")

        if not os.path.exists(graph_path):
            print(f"Graph {i} not found. Skipping.")
            continue

        with open(graph_path, "rb") as f:
            grafo_0 = pickle.load(f)

        graph_list.append(grafo_0)

    return graph_list


def triu_to_dense(triu_values, num_nodes, device):
    """
    Converts a triangular upper part of a matrix as flat vector
    to a squared adjacency matrix with a specific size (num_nodes).
    """
    dense_adj = torch.zeros((num_nodes, num_nodes)).to(device).float()
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)
    dense_adj[triu_indices[0], triu_indices[1]] = triu_values
    dense_adj[tril_indices[0], tril_indices[1]] = triu_values
    return dense_adj


def triu_to_3d_dense(triu_values, num_nodes, device, depth=len(SUPPORTED_EDGES)):
    """
    Converts the triangular upper part of a matrix
    for several dimensions into a 3d tensor.
    """
    # Create placeholder for 3d matrix
    adj_matrix_3d = torch.empty((num_nodes, num_nodes, depth), dtype=torch.float, device=device)
    for edge_type in range(len(SUPPORTED_EDGES)):
        adj_mat_edge_type = triu_to_dense(triu_values[:, edge_type].float(), num_nodes, device=device)
        adj_matrix_3d[:, :, edge_type] = adj_mat_edge_type

    return adj_matrix_3d


def adj_to_edge_index(matrice_adiacenza):
    # Assicuriamoci che la matrice sia un tensore PyTorch
    if not isinstance(matrice_adiacenza, torch.Tensor):
        matrice_adiacenza = torch.tensor(matrice_adiacenza)

    # Trova le posizioni degli elementi non zero in ogni matrice del batch
    indici_non_zero = torch.nonzero(matrice_adiacenza, as_tuple=True)

    # Crea l'edge_index per ogni matrice nel batch
    edge_index = torch.stack([indici_non_zero[1], indici_non_zero[2]])

    # Crea un tensore che indica a quale grafo del batch appartiene ogni arco
    batch_index = indici_non_zero[0]

    return edge_index, batch_index


def recover_adj_lower(vector, max_num_nodes):
    """
    Recover adjacency upper triangular matrix from vector
    """
    rows, _ = vector.size()

    # Creare una matrice con gli zeri
    adj = torch.zeros(rows, max_num_nodes, max_num_nodes, device=vector.device)
    adj[torch.triu(torch.ones(rows, max_num_nodes, max_num_nodes)) == 1] = vector.view(1, -1)

    return adj


def recover_full_adj_from_lower(lower):
    """
    Recover the full adjacency matrix from the lower triangular part.

    Args:
        self: the object instance
        lower: the lower triangular matrix

    Returns:
        batch_matrici_diagonali: the recovered full adjacency matrix
    """
    batch_matrici_diagonali = (
        lower + lower.transpose(-2, -1) - torch.diag_embed(lower.diagonal(dim1=-2, dim2=-1))
    )
    return batch_matrici_diagonali

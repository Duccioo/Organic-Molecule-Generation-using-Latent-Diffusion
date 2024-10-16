import os
import networkx as nx
import numpy as np
import torch
from torch.utils.data import random_split


from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

import rdkit.Chem as Chem

# ---
from utils.graph_utils import matrix_graph_to_nx


class ToTensor(BaseTransform):
    def __call__(self, data):
        data.y = data.y.clone().detach()
        data.adj = torch.tensor(data.adj)
        return data


class CostumPad(BaseTransform):
    def __init__(self, max_num_nodes, max_num_edges):
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

    def __call__(self, data):
        data.adj = pad_adjacency_matrix(data.adj, self.max_num_nodes)
        data.edge_attr = pad_features(data.edge_attr, self.max_num_edges)
        data.x = pad_features(data.x, self.max_num_nodes)
        data.num_nodes = self.max_num_nodes
        data.num_edges = self.max_num_edges

        return data


class AddAdj(BaseTransform):
    def __call__(self, data):
        # Aggiungere gli adiacenti

        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        if edge_list:
            G = nx.Graph(edge_list)
            adj = nx.adjacency_matrix(G).todense()

        # edge_index è la lista di adiacenza del grafo
        # num_nodes è il numero totale di nodi nel grafo
        # adj_matrix = to_dense_adj(edge_index)

        data.adj = adj

        return data


class OneHotEncoding(BaseTransform):
    def __call__(self, data):
        # Indice della colonna da codificare
        col_index = 5

        # Applichiamo one-hot encoding utilizzando la funzione di numpy
        # Dizionario di mapping specificato
        mapping_dict = {
            1: [0, 0, 0, 0],
            6: [1, 0, 0, 0],
            7: [0, 1, 0, 0],
            8: [0, 0, 1, 0],
            9: [0, 0, 0, 1],
        }

        data.x = one_hot_encoding(data.x, col_index, mapping_dict)

        col_index = 13
        mapping_dict = {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1],
            4: [0, 0, 0, 0],
        }

        data.x = one_hot_encoding(data.x, col_index, mapping_dict)

        return data


class ToNx(BaseTransform):
    def __call__(self, data):

        G = matrix_graph_to_nx(data.adj, data.x, data.edge_attr)
        data.G = G

        return data


def one_hot_encoding(matrix, col_index, mapping_dict):

    # One Hot Encoding

    # Otteniamo la colonna da codificare
    col_to_encode = matrix[:, col_index]
    # Applichiamo il mapping utilizzando il metodo map di Python
    one_hot_encoded = torch.tensor(list(map(lambda x: mapping_dict[int(x.item())], col_to_encode)))

    # Sostituiamo la quinta colonna con la codifica one-hot
    matrix = torch.cat(
        (
            matrix[:, :col_index],
            one_hot_encoded,
            matrix[:, col_index + 1 :],
        ),
        dim=1,
    )

    return matrix


class FilterSingleton(BaseTransform):
    def __call__(self, data):

        len_iniziale = len(data.x)
        data = remove_hydrogen(data)
        len_dopo = len(data.x)

        if data.x.size(0) == 1 or data == None:
            print(data.x)
            print("---", len_iniziale, len_dopo)

            return False
        else:

            return True


class FilterMaxNodes(BaseTransform):
    def __init__(self, max_num_nodes):
        self.max_num_nodes = max_num_nodes

    def __call__(self, data):
        return self.max_num_nodes == -1 or data.num_nodes <= self.max_num_nodes


def remove_hydrogen(data):

    # Verifica se l'atomo è un idrogeno (Z=1)
    is_hydrogen = data.z == 1

    # Filtra gli edge e gli edge_index
    edge_mask = ~is_hydrogen[data.edge_index[0]] & ~is_hydrogen[data.edge_index[1]]
    data.edge_index = data.edge_index[:, edge_mask]

    # Aggiorna gli indici degli atomi in edge_index
    _, new_indices = torch.unique(data.edge_index, return_inverse=True)
    data.edge_index = new_indices.reshape(data.edge_index.shape)

    # Filtra le features dei nodi
    data.x = data.x[~is_hydrogen]
    data.edge_attr = data.edge_attr[edge_mask]

    data.z = data.z[~is_hydrogen]

    data.num_nodes = len(data.x)
    data.num_edges = len(data.edge_attr)

    return data


def pad_adjacency_matrix(adj_matrix, target_size):
    padded_adj_matrix = torch.nn.functional.pad(
        adj_matrix,
        (0, target_size - adj_matrix.size(0), 0, target_size - adj_matrix.size(1)),
    )
    return padded_adj_matrix


def pad_features(features_matrix, target_size):
    padded_features = torch.nn.functional.pad(
        features_matrix, (0, 0, 0, target_size - features_matrix.size(0))
    )
    return padded_features


def create_padded_graph_list(
    dataset,
    max_num_nodes_padded=-1,
    remove_duplicates_features: bool = True,
):

    max_num_nodes = max_num_nodes_padded
    # max_num_edges = max([data.num_edges for data in dataset])

    # numero massimo teorico per un graph
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    graph_list = []
    for data in dataset:

        if remove_duplicates_features:
            # rimuovi duplicati dalla matrice delle features degli edges
            # cerca gli edge unici
            unique_edges = list(set(tuple(sorted(x.tolist())) for x in data.edge_index.T))
            # estraggo gli indici degli edge unici
            indices = [data.edge_index.T.tolist().index(list(edge)) for edge in unique_edges]
            data.edge_attr_removed_duplicates = data.edge_attr[indices]
        else:
            data.edge_attr_removed_duplicates = data.edge_attr

        data.adj = pad_adjacency_matrix(data.adj, max_num_nodes)
        data.edge_attr_removed_duplicates = pad_features(data.edge_attr_removed_duplicates, max_num_edges)
        data.x = pad_features(data.x, max_num_nodes)  # features dei nodi
        # data.edge_index = pad_features(data.edge_index.T, max_num_edges).T
        # data.num_edges = data.num_edges.tolist()[0]
        data.num_nodes = len(data.z)
        data.edge_attr = data.edge_attr  # features degli edges

        graph = {
            "adj": np.array(data.adj, dtype=np.float32),
            "features_nodes": data.x,  # Aggiunta delle features con padding
            "features_edges": data.edge_attr_removed_duplicates,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr_removed_duplicates,
            "num_nodes": len(data.z),
            "num_edges": data.num_edges.tolist()[0],
            "smiles": data.smiles,
        }

        graph_list.append(graph)

    return dataset, graph_list, max_num_nodes, max_num_edges


def smiles_to_graph(smiles):
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features


def graph_to_molecule(graph):
    # Unpack graph
    adjacency, features = graph

    # RWMol is a molecule object intended to be edited
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis=1) != ATOM_DIM - 1) & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for bond_ij, atom_i, atom_j in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule


def load_QM9_metric(path, max_num_nodes):
    dataset = QM9(
        root=path,
        pre_filter=T.ComposeFilters([FilterSingleton(), FilterMaxNodes(max_num_nodes)]),
        pre_transform=T.Compose([AddAdj(), ToNx()]),
        transform=T.Compose([ToTensor()]),
    )
    return dataset


def load_QM9_dataloader(
    data_path: str = "QM9",
    max_num_nodes: int = 6,
    num_examples: int = 1000,
    batch_size: int = 5,
    dataset_split_list: tuple = (0.7, 0.2, 0.1),
    apriori_max_num_nodes: int = -1,
    num_workers: int = 0,
    validation_batch_multiplier: int = 1,
):

    # loading dataset
    dataset = QM9(
        root=data_path,
        pre_filter=T.ComposeFilters([FilterSingleton(), FilterMaxNodes(apriori_max_num_nodes)]),
        pre_transform=T.Compose([OneHotEncoding(), AddAdj()]),
        transform=T.Compose([ToTensor()]),
    )

    num_graphs_raw = len(dataset)
    print("Number of graphs raw: ", num_graphs_raw)

    # Filtra i grafi con un numero di nodi maggiore di max_num_nodes
    # dataset = [data for data in dataset if data.num_nodes <= max_num_nodes]
    dataset = dataset[0:num_examples]

    dataset = list(dataset)
    print("dataset loaded!")
    # casualmente prendo un certo numero di grafi dal dataset:
    # dataset = random.sample(dataset, num_examples)
    print("Number of graphs: ", len(dataset))

    max_num_nodes_dataset = max([dataset[i].num_nodes for i in range(len(dataset))])

    dataset, dataset_padded, max_num_nodes, _ = create_padded_graph_list(dataset, max_num_nodes)

    # split dataset
    train_size = int(dataset_split_list[0] * len(dataset))

    test_size = int(dataset_split_list[1] * len(dataset))
    val_size = len(dataset) - train_size - test_size

    # val_size = int(dataset_split_list[2] * len(dataset))
    # print(val_size, train_size, test_size)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train_dataset_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    # )

    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    val_dataset_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=validation_batch_multiplier * batch_size,
        num_workers=num_workers,
    )

    return (
        dataset,
        dataset_padded,
        train_dataset_loader,
        test_dataset_loader,
        val_dataset_loader,
        max_num_nodes_dataset,
    )


if __name__ == "__main__":

    """
    ### Define helper functions
    These helper functions will help convert SMILES to graphs and graphs to molecule objects.

    **Representing a molecular graph**. Molecules can naturally be expressed as undirected
    graphs `G = (V, E)`, where `V` is a set of vertices (atoms), and `E` a set of edges
    (bonds). As for this implementation, each graph (molecule) will be represented as an
    adjacency tensor `A`, which encodes existence/non-existence of atom-pairs with their
    one-hot encoded bond types stretching an extra dimension, and a feature tensor `H`, which
    for each atom, one-hot encodes its atom type. Notice, as hydrogen atoms can be inferred by
    RDKit, hydrogen atoms are excluded from `A` and `H` for easier modeling.

    """

    atom_mapping = {"C": 0, 0: "C", "N": 1, 1: "N", "O": 2, 2: "O", "F": 3, 3: "F"}

    bond_mapping = {
        "SINGLE": 0,
        0: Chem.BondType.SINGLE,
        "DOUBLE": 1,
        1: Chem.BondType.DOUBLE,
        "TRIPLE": 2,
        2: Chem.BondType.TRIPLE,
        "AROMATIC": 3,
        3: Chem.BondType.AROMATIC,
    }

    NUM_ATOMS = 9  # Maximum number of atoms
    ATOM_DIM = 4 + 1  # Number of atom types
    BOND_DIM = 4 + 1  # Number of bond types
    LATENT_DIM = 64  # Size of the latent space

    # Test helper functions
    graph_to_molecule(smiles_to_graph(smiles))

    """
    ### Generate training set

    To save training time, we'll only use a tenth of the QM9 dataset.
    """

    adjacency_tensor, feature_tensor = [], []
    for smiles in data[::10]:
        adjacency, features = smiles_to_graph(smiles)
        adjacency_tensor.append(adjacency)
        feature_tensor.append(features)

    adjacency_tensor = np.array(adjacency_tensor)
    feature_tensor = np.array(feature_tensor)

    print("adjacency_tensor.shape =", adjacency_tensor.shape)
    print("feature_tensor.shape =", feature_tensor.shape)

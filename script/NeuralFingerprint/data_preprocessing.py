from torch_geometric.transforms import BaseTransform
import torch
import networkx as nx

from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class ToTensor(BaseTransform):
    def __call__(self, data):
        # data.y = data.y.clone().detach()
        # data.adj = torch.tensor(data.adj)
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
            # print(data.x)
            # print("---", len_iniziale, len_dopo)

            return False
        else:

            return True


def generate_fingerprint_from_smiles(smiles, size=2048, radius=3):
    """
    Generate a fingerprint from a smiles string.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.
    size : int, optional
        The size of the fingerprint. The default is 2048.

    Returns
    -------
    fingerprint : DataStructs.ExplicitBitVect
        The Morgan fingerprint of the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # fingerprint = list(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=size))
    # fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    fingerprint = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=size)

    return fingerprint.GetFingerprintAsNumPy(mol)


class FilterErrorMolecula(BaseTransform):
    def __call__(self, data):
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            return False

        return True


class AddFIngerprint(BaseTransform):

    def __init__(self, size_fingerprint):
        self.size_fingerprint = size_fingerprint

    def __call__(self, data):
        # è questo che da fastidio perchè la prima dimensione cambia in funzione di quanti sono gli atomi dioOOOOOOoo!@!!!!!
        del data.adj

        rdkit_fp = generate_fingerprint_from_smiles(data.smiles, self.size_fingerprint)
        if rdkit_fp is None:
            data.y = None
            return data

        data.y = torch.tensor(rdkit_fp, dtype=torch.float)

        return data


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


def from_onehot_to_atom(data, index_from: int, index_to: int):

    mapping = {0: 6, 1: 7, 2: 8, 3: 9}
    # mapping = {0: "C", 1: "O", 2: "N", 3: "F"}
    data_decoded = torch.argmax(data[:, index_from:index_to], dim=1)
    data_decoded = [mapping[x.item()] for x in data_decoded]
    return data_decoded


def qm9_to_rdkit(data):
    # Estrai le informazioni
    atom_features = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    # Crea la molecola RDKit
    mol = Chem.RWMol()

    # print("DEBUG: ", atom_features.shape, edge_index.shape, edge_attr.shape)
    # print("SMILE", data.smiles)
    # print("PRE decodifica", edge_attr)
    data_decoded = from_onehot_to_atom(data.x, 5, 9)
    # print("POST decodifica", data_decoded)

    # Aggiungi gli atomi
    atom_mapping = {}

    for i, features in enumerate(data_decoded):
        atom_type = features  # Assumiamo che il tipo di atomo sia il primo feature
        atom = Chem.Atom(atom_type)
        atom_idx = mol.AddAtom(atom)
        atom_mapping[i] = atom_idx

    # Funzione per convertire la codifica one-hot in tipo di legame RDKit
    def get_bond_type(one_hot):
        if torch.equal(one_hot, torch.tensor([1, 0, 0, 0])):
            return Chem.BondType.SINGLE
        elif torch.equal(one_hot, torch.tensor([0, 1, 0, 0])):
            return Chem.BondType.DOUBLE
        elif torch.equal(one_hot, torch.tensor([0, 0, 1, 0])):
            return Chem.BondType.TRIPLE
        elif torch.equal(one_hot, torch.tensor([0, 0, 0, 1])):
            return Chem.BondType.AROMATIC
        else:
            raise ValueError("Tipo di legame non riconosciuto")

    # Aggiungi i legami
    # print("EDGE INDEX", edge_index)
    added_bonds = set()
    for i in range(edge_index.shape[1]):
        start, end = edge_index[:, i]
        start, end = min(start.item(), end.item()), max(start.item(), end.item())
        if (start, end) not in added_bonds:
            # print("START, END", start, end)
            bond_type = get_bond_type(edge_attr[i])
            print("--->", atom_mapping)
            print("--___---", data_decoded)
            print("------------->", edge_index[0])
            print("------------->", edge_index[1])
            print("ADDING BOND", start, end, bond_type)
            try:
                mol.AddBond(atom_mapping[start], atom_mapping[end], bond_type)
            except:
                print("ERROR Impossibile aggiungere legame")

                mol.AddBond(atom_mapping[start], atom_mapping[end - 1], bond_type)
            added_bonds.add((start, end))

    # Converti in molecola RDKit
    mol = mol.GetMol()
    # print("MOLECOLA CONVERTITA")

    # Aggiungi idrogeni impliciti e calcola le coordinate 3D
    # mol = Chem.AddHs(mol)
    # AllChem.EmbedMolecule(mol, randomSeed=42)
    flag = Chem.SanitizeMol(mol, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return mol

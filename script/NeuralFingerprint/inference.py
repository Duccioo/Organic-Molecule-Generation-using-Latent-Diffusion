import torch
from torch_geometric.nn import NeuralFingerprint
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
import os

# ---
from data_preprocessing import FilterSingleton, FilterMaxNodes, OneHotEncoding, AddAdj, ToTensor, qm9_to_rdkit


def load_best_model(path, num_features, hidden_channels=64, num_layers=3, out_channels=1024, device="cpu"):
    model = NeuralFingerprint(num_features, hidden_channels, out_channels, num_layers)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def generate_fingerprint_from_smiles(smiles, size=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # fingerprint = list(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=size))
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size)

    return fingerprint.GetFingerprint(mol)


def molecule_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Ottieni le feature degli atomi (potrebbe richiedere personalizzazione)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
        ]
        atom_features.append(features)

    # Crea la matrice di adiacenza
    adjacency = Chem.GetAdjacencyMatrix(mol)

    # Ottieni gli attributi dei legami (potrebbe richiedere personalizzazione)
    edge_attr = []
    for bond in mol.GetBonds():
        features = [bond.GetBondTypeAsDouble(), bond.GetIsConjugated(), bond.IsInRing()]
        edge_attr.append(features)
        edge_attr.append(features)  # aggiungi due volte per entrambe le direzioni

    # Converti in tensori PyTorch
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(np.array(adjacency.nonzero()).T, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def generate_model_fingerprint(model, graph):
    with torch.no_grad():
        print(graph.x.shape)
        print(graph.edge_index.shape)
        print(graph.batch.shape)
        print(graph.edge_attr.shape)
        return model(graph.x, graph.edge_index, graph.batch).numpy()


def load_data(root="path/to/data"):
    return QM9(
        root=root,
        pre_filter=T.ComposeFilters([FilterSingleton(), FilterMaxNodes(9)]),
        pre_transform=T.Compose([OneHotEncoding(), AddAdj()]),
        transform=T.Compose([ToTensor()]),
    )


def stampa_posizioni_non_zero(vettore):
    counter = 0
    for indice, valore in enumerate(vettore):
        if valore != 0:
            print(f"Posizione {indice}: {valore}")
            counter += 1
    return counter


def main():
    # Carica e prepara i dati
    path_data_qm9 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "QM9")
    dataset = load_data(path_data_qm9)
    print(dataset.num_features)

    # Carica il modello migliore
    model_path = "models/best_fingerprint_model_fold.pth" 
    model = load_best_model(model_path, dataset.num_features)

    # SMILES di esempio
    graph = dataset[np.random.randint(0, len(dataset))]

    # Genera il fingerprint RDKit
    rdkit_fp = generate_fingerprint_from_smiles(graph.smiles)

    # Aggiungi un batch dimension (necessario per l'inferenza)
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

    # Genera il fingerprint dal modello
    model_fp = generate_model_fingerprint(model, graph)

    # Stampa i risultati
    print("RDKit Fingerprint:")
    print(np.array(rdkit_fp))
    counter_vero = stampa_posizioni_non_zero(np.array(rdkit_fp))
    print(counter_vero)
    print("\nModel Generated Fingerprint:")
    print(np.round(model_fp.flatten()))  # Appiattisci l'output se necessario
    counter_falso = stampa_posizioni_non_zero(np.array(np.round(model_fp.flatten())))
    print(counter_falso)

    # Calcola e stampa alcune metriche di similarità
    similarity = np.dot(rdkit_fp, model_fp.flatten()) / (np.linalg.norm(rdkit_fp) * np.linalg.norm(model_fp))
    mse = np.mean((rdkit_fp - model_fp.flatten()) ** 2)

    print(f"\nCosine Similarity: {similarity:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")


if __name__ == "__main__":
    main()

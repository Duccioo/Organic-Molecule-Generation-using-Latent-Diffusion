DEVICE = "cpu"

# Supported edge types
SUPPORTED_EDGES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# Supported atoms
SUPPORTED_ATOMS = ["C", "N", "O", "F"]
ATOMIC_NUMBERS = [6, 7, 8, 9]

# Dataset (if you change this, delete the processed files to run again)
MAX_MOLECULE_SIZE = 9

# Numero massimo di edge teorico per un grafo dato un numero "MAX_MOLECULE_SIZE" di atomi
MAX_EDGE_SIZE = MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1) // 2

# To remove valence errors ect.
DISABLE_RDKIT_WARNINGS = True

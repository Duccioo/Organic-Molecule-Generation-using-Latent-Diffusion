import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
import numpy as np


def reduce_dimensions(matrix, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(matrix)


def create_graph(adj_matrix, node_features, edge_features):
    G = nx.from_numpy_array(adj_matrix)

    # Aggiungi attributi ai nodi
    for i, features in enumerate(node_features):
        G.nodes[i]["features"] = features

    # Aggiungi attributi agli archi
    for u, v, d in G.edges(data=True):
        d["features"] = edge_features[u][v]

    return G


def plot_graphs(graph_dict, plot_type):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{plot_type} Graphs", fontsize=16)

    for i, (key, value_list) in enumerate(
        [
            (f"{plot_type}_adj", graph_dict[f"{plot_type}_adj"]),
            (f"{plot_type}_node", graph_dict[f"{plot_type}_node"]),
            (f"{plot_type}_edge", graph_dict[f"{plot_type}_edge"]),
        ]
    ):
        ax = axes[i]
        ax.set_title(key)

        if key.endswith("_adj"):
            # Visualizza la matrice di adiacenza
            im = ax.imshow(value_list[0], cmap="viridis")
            plt.colorbar(im, ax=ax)
        else:
            # Riduci le dimensioni e visualizza come scatter plot
            reduced_data = reduce_dimensions(np.array(value_list))
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.show()


def visualize_graphs(train_dict):
    # Plot per gli "example"
    plot_graphs(train_dict, "example")

    # Plot per le "recon"
    plot_graphs(train_dict, "recon")


if __name__ == "__main__":
    # Esempio di utilizzo
    train_dict_example_and_recon = {
        "example_adj": [np.random.rand(10, 10) for _ in range(5)],
        "example_node": [np.random.rand(10, 5) for _ in range(5)],
        "example_edge": [np.random.rand(10, 10, 3) for _ in range(5)],
        "example_smiles": ["C1=CC=CC=C1", "CC(=O)O", "C1CCCCC1", "CCO", "CN"],
        "recon_adj_vec": [np.random.rand(10, 10) for _ in range(5)],
        "recon_node": [np.random.rand(10, 5) for _ in range(5)],
        "recon_edge": [np.random.rand(10, 10, 3) for _ in range(5)],
    }

    visualize_graphs(train_dict_example_and_recon)

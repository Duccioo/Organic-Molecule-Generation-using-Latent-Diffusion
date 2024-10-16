import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from matplotlib import pyplot as plt

# from GNN.composite_graph_class import CompositeGraphObject
# from GNN.graph_class import GraphObject

# uno script con la funzione di riordinamento dei nodi.


#######################################################################################################################
# OPTIONS #############################################################################################################
#######################################################################################################################


#######################################################################################################################
# FUNCTIONS ###########################################################################################################
#######################################################################################################################
def reorder_graph(directed_graph, label_int_order, use_bfs: bool = True, return_DF: bool = False):
    # mi salvo il nome delle chiavi delle features dei nodi e degli edge
    node_features = [list(node[1].keys()) for node in directed_graph.nodes(data=True)][0][0]
    edge_features = [list(edge[2].keys()) for edge in directed_graph.edges(data=True)][0][0]

    # Calculating each node's betwennes
    betweenness_centrality = nx.betweenness_centrality(directed_graph)

    DF = pd.DataFrame.from_dict(betweenness_centrality, orient="index", columns=["centrality"])

    DF["degree"] = dict(directed_graph.degree()).values()
    DF["atom_type"] = [directed_graph.nodes[i][node_features] for i in betweenness_centrality.keys()]

    DF["atom_type_int"] = DF["atom_type"].map(label_int_order)
    DF.sort_values(by=["centrality", "degree", "atom_type_int"], ascending=[False, False, True], inplace=True)
    DF.reset_index(inplace=True)
    DF.rename(columns={"index": "original_index"}, inplace=True)

    ### BFS
    i = 0
    BFS = [DF["original_index"][0]]
    while len(BFS) != len(DF):
        n = list(directed_graph.neighbors(BFS[i]))
        s = list(set(n) - set(BFS))
        s = sorted(s, key=lambda original_idx: DF.loc[DF["original_index"] == original_idx].index)

        BFS += s
        i += 1
    DF["bfs_weighted_order"] = BFS

    # Create a dictionary from the DataFrame columns "node_index" and "new_idx"
    new_order = dict(zip(DF["bfs_weighted_order" if use_bfs else "original_index"], DF.index))

    # Create a new graph with reindexed nodes
    H = nx.Graph()

    # Add nodes to the new graph with the new indexes and labels
    for old_index, new_index in new_order.items():
        H.add_node(new_index, atom_type=directed_graph.nodes[old_index][node_features])

    # Add edges to the new graph with labels (you'll need to adapt this part based on your original graph's structure)
    for edge in directed_graph.edges():
        node1, node2 = edge
        new_node1 = new_order[node1]
        new_node2 = new_order[node2]
        edge_label_bondtype = directed_graph.get_edge_data(node1, node2)[edge_features]
        # edge_label_bondstereo = directed_graph.get_edge_data(node1, node2)["bond_stereo"]
        # H.add_edge(new_node1, new_node2, bond_type=edge_label_bondtype, bond_stereo=edge_label_bondstereo)
        H.add_edge(new_node1, new_node2, bond_type=edge_label_bondtype)

    if return_DF:
        return H.to_directed(), DF
    return H.to_directed()


def reorder_graph_gpu(data, label_int_order, use_bfs=True, return_df=False):
    device = data.x.device

    # Calcoliamo la betweenness centrality
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    row, col = edge_index
    deg = scatter_add(torch.ones(row.size(0), device=device), row, dim=0, dim_size=num_nodes)

    # Approssimazione della betweenness centrality usando il grado
    betweenness = deg / deg.sum()

    # Creiamo un tensore con le informazioni necessarie per l'ordinamento
    sorting_info = torch.stack([betweenness, deg, data.x.squeeze()], dim=1)

    # Ordiniamo i nodi
    _, sorted_indices = torch.sort(sorting_info, dim=0, descending=True)
    new_order = torch.argsort(sorted_indices[:, 0])

    if use_bfs:
        # Implementiamo BFS
        bfs_order = torch.zeros(num_nodes, dtype=torch.long, device=device)
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        queue = [new_order[0].item()]
        visited[new_order[0]] = True
        idx = 0

        while queue:
            current = queue.pop(0)
            bfs_order[idx] = current
            idx += 1
            neighbors = col[row == current]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor.item())

        new_order = bfs_order

    # Riordiniamo i nodi e gli archi
    data.x = data.x[new_order]
    row = torch.tensor([torch.where(new_order == r)[0].item() for r in row], device=device)
    col = torch.tensor([torch.where(new_order == c)[0].item() for c in col], device=device)
    data.edge_index = torch.stack([row, col], dim=0)

    if return_df:
        # Creiamo un DataFrame-like structure usando dizionari e liste
        df = {
            "centrality": betweenness[new_order].cpu().numpy(),
            "degree": deg[new_order].cpu().numpy(),
            "atom_type": data.x.cpu().numpy(),
            "original_index": new_order.cpu().numpy(),
            "new_index": torch.arange(num_nodes, device=device).cpu().numpy(),
        }
        return data, df

    return data


# def reorder_graph(input_graph, label_int_order):
#     #Turning the graph to digraph
#     input_graph = input_graph.to_directed()
#     #Ordering the graph by centrality and atom type
#     H = reorder_cent_type(input_graph, label_int_order)
#
#     #In this new graph, from node 0 (highest centrality, get the DFS visit
#     dfs_ordering = list(nx.dfs_preorder_nodes(H, source=0))
#
#     new_order = {}
#
#     for i in range(len(H)):
#         new_order[i] = dfs_ordering[i]
#
#     new_order = {value: key for key, value in new_order.items()}
#
#     # Create a new graph with reindexed nodes
#     H_dfs = nx.Graph()
#
#     # Add nodes to the new hraph with the new indexes and labels
#     for new_index, old_index in new_order.items():
#         H_dfs.add_node(new_index, atom_type=H.nodes[old_index]['atom_type'])
#
#     # Add edges to the new graph with labels (you'll need to adapt this part based on your original graph's structure)
#     for edge in H.edges():
#         node1, node2 = edge
#         new_node1 = new_order[node1]
#         new_node2 = new_order[node2]
#         edge_label_bondtype = H.get_edge_data(node1, node2)['bond_type']
#         edge_label_bondstereo = H.get_edge_data(node1, node2)["bond_stereo"]
#         H_dfs.add_edge(new_node1, new_node2, bond_type=edge_label_bondtype, bond_stereo=edge_label_bondstereo)
#
#     H_dfs = H_dfs.to_directed()
#
#     return H_dfs
#
#######################################################################################################################
# SCRIPT ##############################################################################################################
#######################################################################################################################


def main() -> None:
    print("Loading Dataset...", end="\t")
    file_list = os.listdir(path)[: int(graph_number)]
    loaded_graphs = list()
    for idx, file in enumerate(file_list):
        if idx % 2000 == 0:
            print(f"Loading Dataset...\t{idx+1}/{len(file_list)}", end="\r")
        with open(f"Zinc250K_all_networkx_graphs/{file}", "rb") as g:
            ordered_g = pickle.load(g).to_directed()
            # ordered_g = reorder_graph(ordered_g, node_importance)
            ordered_g = reorder_graph(ordered_g, node_importance)
            loaded_graphs.append(ordered_g)
    print("Loading and Ordering Dataset...\t", end="\t")

    ### NODES
    print("OK\nLoading Nodes...", end="\t")
    nodes = [pd.DataFrame([g.nodes[k] for k in g.nodes], columns=["atom_type"]) for g in loaded_graphs]
    atom_str_to_int_dict = pd.concat(nodes, axis=0, ignore_index=True)
    atom_str_to_int_dict = dict(
        zip(
            np.unique(atom_str_to_int_dict["atom_type"]),
            range(len(np.unique(atom_str_to_int_dict["atom_type"]))),
        )
    )
    for n in nodes:
        n["atom_type_int"] = n.replace(atom_str_to_int_dict)

    NODES = [np.eye(len(atom_str_to_int_dict))[n["atom_type_int"].values] for n in nodes]

    ### ARCS
    print("OK\nLoading Arcs...", end="\t")
    arcs = [pd.DataFrame(g.edges, columns=["idx_start", "idx_end"]) for g in loaded_graphs]
    arcs = [
        pd.concat([a, pd.DataFrame(g.edges[k] for k in g.edges)], axis=1) for a, g in zip(arcs, loaded_graphs)
    ]

    all_arcs = pd.concat(arcs, axis=0, ignore_index=True)
    replacing_dict = {"bond_type": dict()}  # , "bond_stereo": dict()}
    for col in replacing_dict:
        replacing_dict[col] = dict(zip(np.unique(all_arcs[col]), range(len(np.unique(all_arcs[col])))))
        for a in arcs:
            a[f"{col}_int"] = a[col].replace(replacing_dict[col])

    ARCS = [
        np.concatenate(
            [a[a.columns[:2]].values]
            + [np.eye(len(val))[a[f"{col}_int"]] for col, val in replacing_dict.items()],
            axis=1,
        )
        for a, g in zip(arcs, loaded_graphs)
    ]
    # bidirectional - commented because each graph is transformed to a directed ones (networkx automatically doubles the arcs)
    # ARCS = [np.unique(np.concatenate([a, a[:, [1,0]+list(range(2, a.shape[1]))]], axis=0), axis=0) for a in ARCS]

    # reorder, jut to be sure, since networkx does this operation in .to_directed() procedure
    ARCS = [np.unique(a, axis=0) for a in ARCS]

    print("OK\nGenerating GraphObjects...", end="\t")
    graphs = [GraphObject(nodes=n, arcs=a, targets=None) for n, a in zip(NODES, ARCS)]

    print("OK\nSaving GraphObjects...", end="\t")
    GraphObject.save_dataset_npz(output_path, graphs)

    print("OK\n>>> END OF SCRIPT")


if __name__ == "__main__":

    # main()
    graph_number: int = 1e7

    path: str = "Zinc250K_all_networkx_graphs"
    output_path: str = "Zinc250k.npz"

    node_importance: dict = {"C": 0, "O": 1, "N": 2, "F": 3, "S": 4, "P": 5, "Br": 6, "Cl": 7, "I": 8}
    data = Data(
        x=torch.tensor([[1], [0], [1], [0]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 1], [1, 0, 2, 1, 3, 2, 1, 3]]),
        edge_attr=[1, 1, 2, 2, 1, 1, 2, 2],
    )
    reordered_data = reorder_graph_gpu(data, node_importance)

    # crea un grafo in NX
    G = nx.Graph()
    G.add_node(0, atom_type="O")
    G.add_node(1, atom_type="C")
    G.add_node(2, atom_type="O")
    G.add_node(3, atom_type="C")
    G.add_edge(0, 1, bond_type=0)
    G.add_edge(1, 2, bond_type=1)
    G.add_edge(2, 3, bond_type=2)
    G.add_edge(1, 3, bond_type=3)

    ordered_g = reorder_graph(G, node_importance)

    exit()
    # plotta i 2 grafi in una sola immagine
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    nx.draw(G, with_labels=True, ax=ax1)
    ax1.set_title("originale")
    nx.draw(ordered_g, with_labels=True, ax=ax2)
    ax2.set_title("ordinato")
    plt.show()

    # # show the difference
    # plt.figure(figsize=(10, 10))

    # nx.draw(ordered_g, with_labels=True)
    # nx.draw(G, with_labels=True)

    # plt.show()

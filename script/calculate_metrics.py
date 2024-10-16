# coding=utf-8
import sys
import os
import numpy as np
import networkx as nx
import pickle
from matplotlib import pyplot as plt
from rdkit import Chem
import argparse

from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from molvs import validate as mv


from tqdm import tqdm
import pathlib


# ---
from utils import check_base_dir, matrix_graph_to_nx, load_QM9_metric
from metric.calculate_ground_truth_metrics import load_array_metrics_from_directory
from metric.plot_distributions import plot_distributions
from metric.calculate_frenchet_distance import frechet_distance
from metric.molecular_metric import calculate_generic_metric

atomic_number = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
atomic_label = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
bond_type = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
# bond_label = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}


# function that returns True if two nodes represent the same atom type
def node_equality(n1, n2):
    return n1["features"] == n2["features"]


# function that returns True if the two edges represent the same bond type
def edge_equality(e1, e2):
    return e1["features"] == e2["features"]


# function that translates a rdkit.Chem.RWMol object to a networkx.Graph object
# def RWMolToNxGraph(molecule):
#     G = nx.Graph()
#     for atom in molecule.GetAtoms():
#         G.add_node(
#             atom.GetIdx(),
#             atomic_num=atom.GetAtomicNum(),
#             formal_charge=atom.GetFormalCharge(),
#             chiral_tag=atom.GetChiralTag(),
#             hybridization=atom.GetHybridization(),
#             num_explicit_hs=atom.GetNumExplicitHs(),
#             is_aromatic=atom.GetIsAromatic(),
#         )
#         G.nodes[atom.GetIdx()]["features"] = atomic_label[atom.GetAtomicNum()]
#     for bond in molecule.GetBonds():
#         G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
#         G.edges[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]["features"] = bond_label[bond.GetBondType()]
#     return G


# validation function that exploits the Molvs validator
def ValidateWithMolvs(molecule):
    validator = mv.Validator()
    error_list = validator.validate(molecule)
    return not error_list


# validation function that exploits the rdkit.Chem.SanitizeMol function
def ValidateWithSanitization(molecule):
    try:
        Chem.SanitizeMol(molecule)
    except:
        return False
    return True


# validation function that exploits an ad-hoc built procedure (the argument should be a networkx.Graph, not a molecule)
def Validate(G):
    # initialize a list of potential validation errors
    error_list = list()
    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        error_list.append("Empty graph")
    # check the valence of every node
    for i, node in enumerate(G.nodes()):
        # retrieve the list of neighbors
        neighbors = list(G.neighbors(node))
        # get atom label
        label = G.nodes[node]["features"]
        # cycle over the neighbors, and calculate the valence of this atom as the sum of the electrons employed in the bonds
        valence = 0
        for j in neighbors:
            valence = valence + G.edges[node, j]["features"]
        # enforce valence rules
        if label == "H":
            if valence != 1:
                error_list.append("Hydrogen atom " + str(i) + " has incorrect valence : " + str(valence))
        elif label == "C":
            if valence != 4:
                error_list.append("Carbon atom " + str(i) + " has incorrect valence : " + str(valence))
        elif label == "N":
            if valence not in [3, 5]:
                error_list.append("Nitrogen atom " + str(i) + " has incorrect valence : " + str(valence))
        elif label == "O":
            if valence != 2:
                error_list.append("Oxygen atom " + str(i) + " has incorrect valence : " + str(valence))
        elif label == "F":
            if valence != 1:
                error_list.append("Fluorine atom " + str(i) + " has incorrect valence : " + str(valence))
    # return the list of errors found in this molecular graph (if the list is empty the molecule is valid)
    return error_list


def save_to_smiles(molecules: list, output_filename: str):
    """
    Save the valid molecules in a file.

    Args:
        molecules (list): A list of valid molecules.
        output_filename (str): The name of the file to write the valid molecules to.
    """
    # prima controllo se il file esiste
    if not os.path.exists(output_filename):
        out_file = open(output_filename, "w")
        out_file.close()

    with open(output_filename, "a") as f:
        for molecule in molecules:
            smiles = Chem.MolToSmiles(molecule)
            f.write(smiles + "\n")
        f.write("-" * 10 + "\n")


def check_Validity(graphs: list, molecules: Chem.RWMol, output_filename: str):
    """
    Check the validity of a list of molecules and their corresponding graphs.

    Args:
        graphs (list): A list of graphs representing the molecules.
        molecules (Chem.RWMol): A list of molecules to be checked for validity.
        output_filename (str): The name of the file to write the validation errors to.

    Returns:
        tuple: A tuple containing two lists. The first list contains the indices of the valid molecules,
        and the second list contains the valid molecules themselves.
    """
    valid_indices = list()
    valid_molecules = list()
    # check for output file
    if not os.path.exists(output_filename):
        out_file = open(output_filename, "w")
        out_file.close()
    progress_bar = tqdm(range(len(molecules)), ascii="░▒▓", desc="Checking validity")
    for i in progress_bar:

        validation_errors = Validate(graphs[i])
        if ValidateWithSanitization(molecules[i]) and not validation_errors:
            valid_indices.append(i)
            valid_molecules.append(molecules[i])
        else:
            out_file = open(output_filename, "a")
            out_file.write("Molecule " + str(i) + " is not valid because:\n")
            for e in validation_errors:
                out_file.write(e + "\n")
            out_file.write("\n")
            out_file.close()
        # progress_bar.set_description("Validity " + str(i + 1) + " of " + str(len(molecules)))
    print(f"\033[F\033[K", end="")  # Cancella l'ultima riga

    progress_bar.close()

    return valid_indices, valid_molecules


def check_Uniqueness(graphs: list, valid_molecules: list, valid_indices: list):
    """
    Check the uniqueness of valid molecules in a list of graphs and their corresponding molecules.

    Args:
        graphs (list): A list of graphs representing the molecules.
        valid_molecules (list): A list of valid molecules.
        valid_indices (list): A list of indices of valid molecules in the `valid_molecules` list.

    Returns:
        tuple: A tuple containing three lists. The first list contains the unique graphs,
        the second list contains the unique valid molecules, and the third list contains the indices
        of the unique valid molecules in the `valid_molecules` list.
    """
    unique_graphs = list()
    unique_molecules = list()
    unique_indices = list()

    progress_bar = tqdm(range(len(valid_indices)), ascii="░▒▓", desc="Checking Uniqueness")
    for i in progress_bar:
        ii = valid_indices[i]
        unique = True
        for j in range(len(unique_graphs)):
            if nx.is_isomorphic(graphs[ii], unique_graphs[j], node_equality, edge_equality):
                unique = False
        if unique:
            unique_graphs.append(graphs[ii])
            unique_molecules.append(valid_molecules[i])
            unique_indices.append(ii)
        # progress_bar.set_description("Uniqueness " + str(i + 1) + " of " + str(len(valid_indices)))
    print(f"\033[F\033[K", end="")  # Cancella l'ultima riga
    progress_bar.close()
    return unique_graphs, unique_molecules, unique_indices


def check_Novelty(
    unique_graphs: list,
    unique_indices: list,
    dataset_size: int,
    training_dataset: str,
    output_filename: str,
):
    """
    Check the novelty of unique molecules against a set of training graphs.

    Args:
        unique_graphs (list): A list of unique graphs representing the unique molecules.
        unique_indices (list): A list of indices of the unique molecules in the `unique_graphs` list.
        dataset_size (int): The size of the training dataset.
        training_folder (str): The path to the folder containing the training graphs.
        output_filename (str): The name of the file to write the results to.

    Returns:
        tuple: A tuple containing two lists. The first list contains the indices of the novel unique molecules,
        and the second list contains the corresponding novel unique graphs.
    """
    novel_graphs = unique_graphs.copy()
    novel_indices = unique_indices.copy()

    # the candidate graphs which survive in the list of novel graphs at the end of this cycle are the true novel graphs
    # progress_bar = tqdm(
    #     enumerate(training_dataset), ascii="░▒▓", desc="Checking Novelty", len=training_dataset
    # )
    for i, training_set_graph in enumerate(training_dataset):
        print(
            "Checking novelty of unique molecules against training graph "
            + str(i + 1)
            + " of "
            + str(dataset_size),
            end="\r",
        )
        # if the list of candidate novel graphs is already empty, break the cycle
        if not novel_graphs:
            break
        # load i-th training set graph
        # screen every candidate novel graph against the i-th training set graph
        drop_index = None
        for j in range(len(novel_graphs)):
            # check if the j-th candidate novel graph is isomorphic to the i-th training set graph
            if nx.is_isomorphic(novel_graphs[j], training_set_graph, node_equality, edge_equality):
                # if drop_index is not None throw an exception, as more than one unique graph is isomorphic to the same graph
                if drop_index is not None:
                    sys.exit(
                        "ERROR: more than one unique graph is isomorphic to graph "
                        + str(i)
                        + " in the training set"
                    )
                # otherwise store j as the index of the non-novel graph to drop
                drop_index = j
        # if a candidate novel graph was found to be a duplicate of the i-th graph in the training set, drop it from the list of candidates
        if drop_index is not None:
            out_file = open(output_filename, "a")
            out_file.write(
                "Generated graph "
                + str(novel_indices[drop_index])
                + " is isomorphic to training graph "
                + str(i)
                + "\n"
            )
            out_file.close()
            novel_graphs.pop(drop_index)
            novel_indices.pop(drop_index)
        # delete the i-th training set graph
        del training_set_graph
    return novel_indices, novel_graphs


def nx_graph_list_to_mol(graphs: list, atomic_numbers: dict, bond_type: dict):
    """
    Converts a list of graphs into a list of RDKit molecules.

    Args:
        graphs (list): A list of NetworkX graphs.

    Returns:
        list: A list of RDKit molecules, where each molecule is a RDKit RWMol object.

    This function iterates over each graph in the input list and translates it into an RDKit molecule.
    It creates an atom object for each node in the graph and adds it to the molecule.
    It then copies the bonds between the nodes and adds them to the molecule.
    Finally, it appends the new molecule object to the list of molecules.

    Note:
        The function assumes that the input graphs have a 'nodes' attribute that contains the node information,
        and an 'edges' attribute that contains the edge information. The node information must have an 'info'
        attribute that specifies the atomic number of the atom. The edge information must have an 'info' attribute
        that specifies the bond type between the two nodes.

    Example:
        >>> import networkx as nx
        >>> g1 = nx.Graph()
        >>> g1.add_nodes_from([1, 2, 3], features=[6, 7, 8])
        >>> g1.add_edges_from([(1, 2), (2, 3)], features=[1, 2])
        >>> graphs = [g1]
        >>> graph_to_mol(graphs)
        [<RDKit RWMol object at 0x7f8735d77e20>]
    """
    molecules = list()

    for i, G in enumerate(graphs):
        print("Translating graph " + str(i + 1) + " of " + str(len(graphs)), end="\r")
        mol = Chem.RWMol()
        # define two dicts to keep track of the atom index corresponding to each node index
        atom_to_node = dict()
        node_to_atom = dict()
        # create an atom object for each node in the graph

        for j, node in enumerate(G.nodes()):

            # print("AAAAAA   ", graphs[i].nodes[nodes].items(), "   AAAAAAAA")
            # print("CARICO IL GRAPHO " + str(i), "con ", graphs[i].nodes[nodes]["features"])

            node_features = G.nodes[node]["features"]

            a = Chem.Atom(atomic_number[atomic_numbers[node_features]])
            a.SetFormalCharge(0)
            index = mol.AddAtom(a)

            atom_to_node[index] = j
            node_to_atom[node] = index

        # copy the bonds
        for e, edge in enumerate(G.edges()):
            # print(e)
            atom0 = node_to_atom[edge[0]]
            atom1 = node_to_atom[edge[1]]
            # print(atom0, atom1)
            try:
                num_label = G.edges[edge[0], edge[1]]["features"] + 1
                if isinstance(num_label, np.ndarray):
                    num_label = num_label[0]
                mol.AddBond(atom0, atom1, bond_type[num_label])
            except:
                print("errore features non trovate")
                # num_label = 1

            # print(f"Adding bond {bond_type[num_label]} between " + str(atom0) + " and " + str(atom1))
        # append the new molecule object to the list
        molecules.append(mol)
    return molecules


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument(
        "--gen_path",
        type=str,
        dest="generated_graph_path",
        help="path to generated graphs",
    )
    parser.add_argument("--model_folder", type=str, help="path to model folder")
    parser.add_argument("--num_samples", type=int, help="Number of samples")

    parser.set_defaults(
        generated_graph_path="generated_graphs_v5_approx",
        model_folder="GraphVAE_v9.1_fingerprint_fs128_50000",
        # model_name = "Diffusion_v9.1_fingerprint_fs128_50000_from_50000"
    )
    return parser.parse_args()


def main():
    atomic_numbers = {0: "C", 1: "N", 2: "O", 3: "F"}
    # bond_type = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}
    bond_type = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    prog_args = arg_parse()

    # setup path and dir
    script_path = os.path.dirname(os.path.realpath(__file__))
    repo_path = os.path.abspath(os.path.join(script_path, os.pardir))
    data_path = check_base_dir(repo_path, "data", "QM9")

    models_folder = os.path.join(repo_path, "models")

    model_name = prog_args.model_folder

    # set the folder paths
    model_folder = os.path.join(repo_path, models_folder, model_name)
    generated_folder = os.path.join(model_folder, "generated_graphs")

    # generated_graph_list = load_graphs_from_folder(folder_path=generated_folder, create_using=nx.Graph)

    with open(os.path.join(generated_folder, "graphs_generated.pkl"), "rb") as f:
        generated_graph_list = pickle.load(f)

    dataset_size = (
        prog_args.num_samples
        if prog_args.num_samples is not None and prog_args.num_samples <= len(generated_graph_list)
        else len(generated_graph_list)
    )
    generated_graph_list = generated_graph_list[0:dataset_size]
    output_folder = check_base_dir(model_folder, f"metrics_{dataset_size}")
    output_folder = pathlib.Path(output_folder)
    output_stats_folder = check_base_dir(model_folder, f"metrics_{dataset_size}", "stats")
    output_filename_summary = os.path.join(output_folder, "summary_metrics.txt")
    output_filename_validity = os.path.join(output_folder, "validity_metrics.txt")
    output_filename_uniqueness = os.path.join(output_folder, "uniqueness_metrics.txt")
    output_filename_novelty = os.path.join(output_folder, "novelty_metrics.txt")

    print("Dataset size: " + str(dataset_size))

    dataset_qm9 = load_QM9_metric(path=data_path, max_num_nodes=9)
    training_graphs_list = []
    for data in tqdm(dataset_qm9[0:dataset_size], desc="Loading training graphs"):
        graph = matrix_graph_to_nx(data.adj, data.x, data.edge_attr)
        training_graphs_list.append(graph)

    metric_folder_dataset = pathlib.Path(data_path, f"metrics_{dataset_size}", "stats")
    metric_generic_dataset = load_array_metrics_from_directory(
        metric_folder_dataset, dataset_size=dataset_size
    )

    # dataset_size = 1
    out_file = open(output_filename_summary, "w")
    out_file.write(f"Evaluation of metrics over generation run n")
    out_file.close()

    # translate each graph in a rdkit.Chem.RWMol object
    molecules = nx_graph_list_to_mol(
        graphs=generated_graph_list, atomic_numbers=atomic_numbers, bond_type=bond_type
    )
    print("")

    # check the chemical validity of each molecule
    valid_indices, valid_molecules = check_Validity(generated_graph_list, molecules, output_filename_validity)
    print("")
    print(
        "Valid molecules    : " + str(len(valid_molecules)) + " / " + str(len(generated_graph_list)),
        f"-> {100 * len(valid_molecules) / len(generated_graph_list):.2f} %",
    )
    print("")

    # check uniqueness of the valid molecules
    unique_graphs, unique_molecules, unique_indices = check_Uniqueness(
        generated_graph_list, valid_molecules, valid_indices
    )
    print(
        "Unique molecules    : "
        + str(len(unique_molecules))
        + " / "
        + str(len(generated_graph_list))
        + f"-> {100 * float(len(unique_molecules) /len(valid_molecules)):.2f}% "
        + f"({100 * float(len(unique_molecules) / len(generated_graph_list)):.2f}% overtotal)",
    )
    print("")

    # check novelty of the unique molecules
    novel_indices, novel_graphs = check_Novelty(
        unique_graphs,
        unique_indices,
        dataset_size,
        training_graphs_list,
        output_filename_novelty,
    )
    print(
        "Novel molecules    : "
        + str(len(novel_graphs))
        + " / "
        + str(len(generated_graph_list))
        + f"-> {100 * len(novel_graphs) / len(unique_graphs):.2f}%"
        + f" ({100 * float(len(novel_graphs)) / float(len(generated_graph_list)):.2f}% overtotal)",
    )
    print("")

    # calculate validity/uniqueness/novelty results
    valid = float(len(valid_molecules)) / float(len(generated_graph_list))
    if valid_molecules:
        unique = float(len(unique_molecules)) / float(len(valid_molecules))
    else:
        unique = 0.0
    if unique_graphs:
        novel = float(len(novel_graphs)) / float(len(unique_graphs))
    else:
        novel = 0.0
    over_total_valid = float(len(valid_molecules)) / float(len(generated_graph_list))
    over_total_unique = float(len(unique_molecules)) / float(len(generated_graph_list))
    over_total_novel = float(len(novel_graphs)) / float(len(generated_graph_list))

    # calculate chemical descriptors of each valid molecule
    # QED_scores = list()
    # logP_scores = list()
    # molecular_weights = list()
    # ring_counts = list()

    metric_generic_generated = calculate_generic_metric(valid_molecules)

    # save vectors of statistics to files for later use
    np.savetxt(
        os.path.join(output_stats_folder, "QED_scores.txt"),
        metric_generic_generated.QED_scores,
        delimiter=" , ",
    )
    np.savetxt(
        os.path.join(output_stats_folder, "logP_scores.txt"),
        metric_generic_generated.logP_scores,
        delimiter=" , ",
    )
    np.savetxt(
        os.path.join(output_stats_folder, "molecular_weights.txt"),
        metric_generic_generated.molecular_weights,
        delimiter=" , ",
    )
    np.savetxt(
        os.path.join(output_stats_folder, "ring_counts.txt"),
        metric_generic_generated.ring_counts,
        delimiter=" , ",
    )

    # print results to file
    print("Printing results to output file : " + output_filename_summary)
    out_file = open(output_filename_summary, "a")
    out_file.write("Molecules Generated : " + str(len(generated_graph_list)) + "\n")
    out_file.write("Valid Molecules : " + str(len(valid_molecules)) + "\n")
    out_file.write("Unique Molecules : " + str(len(unique_molecules)) + "\n")
    out_file.write("Novel Molecules : " + str(len(novel_graphs)) + "\n")
    out_file.write("Valid Fraction : " + str(valid) + "\n")
    out_file.write("Unique Fraction : " + str(unique) + "\n")
    out_file.write("Novel Fraction : " + str(novel) + "\n")
    out_file.write("Valid Over Total : " + str(over_total_valid) + "\n")
    out_file.write("Unique Over Total : " + str(over_total_unique) + "\n")
    out_file.write("Novel Over Total : " + str(over_total_novel) + "\n")
    out_file.write(
        "Average QED (stdev): "
        + str(metric_generic_generated.QED_scores_avg)
        + " ("
        + str(metric_generic_generated.QED_scores_std)
        + ")\n"
    )
    out_file.write(
        "Average logP (stdev) : "
        + str(metric_generic_generated.logP_scores_avg)
        + " ("
        + str(metric_generic_generated.logP_scores_std)
        + ")\n"
    )
    out_file.write(
        "Average Mol.Wt. (stdev) : "
        + str(metric_generic_generated.molecular_weights_avg)
        + " ("
        + str(metric_generic_generated.molecular_weights_std)
        + ")\n"
    )
    out_file.write(
        "Average Ring C. (stdev) : "
        + str(metric_generic_generated.ring_counts_avg)
        + " ("
        + str(metric_generic_generated.ring_counts_std)
        + ")\n"
    )
    print(metric_generic_dataset)
    frenchet_dist_qm9 = frechet_distance(
        metric_generic_generated.QED_scores, metric_generic_dataset.QED_scores.reshape(1, -1)
    )
    out_file.write("Frenchet Dist. QM9: " + str(frenchet_dist_qm9) + "\n")

    frenchet_dist_logP = frechet_distance(
        metric_generic_generated.logP_scores, metric_generic_dataset.logP_scores.reshape(1, -1)
    )
    out_file.write("Frenchet Dist. LogP: " + str(frenchet_dist_logP) + "\n")

    frenchet_dist_mw = frechet_distance(
        metric_generic_generated.molecular_weights, metric_generic_dataset.molecular_weights.reshape(1, -1)
    )
    out_file.write("Frenchet Dist. molecular weights: " + str(frenchet_dist_mw) + "\n")

    frenchet_dist_ring = frechet_distance(
        metric_generic_generated.ring_counts, metric_generic_dataset.ring_counts.reshape(1, -1)
    )
    out_file.write("Frenchet Dist. Ring: " + str(frenchet_dist_ring) + "\n")

    out_file.close()

    # print results to screen
    print("Molecules Generated : " + str(len(generated_graph_list)))
    print("Valid Molecules : " + str(len(valid_molecules)))
    print("Unique Molecules : " + str(len(unique_molecules)))
    print("Novel Molecules : " + str(len(novel_graphs)))
    print("Valid Fraction : " + str(valid))
    print("Unique Fraction : " + str(unique))
    print("Novel Fraction : " + str(novel))
    print("Valid Over Total : " + str(over_total_valid))
    print("Unique Over Total : " + str(over_total_unique))
    print("Novel Over Total : " + str(over_total_novel))
    print(
        "Average QED (stdev) : "
        + str(metric_generic_generated.QED_scores_avg)
        + " ("
        + str(metric_generic_generated.QED_scores_std)
        + ")"
    )
    print(
        "Average logP (stdev) : "
        + str(metric_generic_generated.logP_scores_avg)
        + " ("
        + str(metric_generic_generated.logP_scores_std)
        + ")"
    )
    print(
        "Average Mol.Wt. (stdev) : "
        + str(metric_generic_generated.molecular_weights_avg)
        + " ("
        + str(metric_generic_generated.molecular_weights_std)
        + ")"
    )
    print(
        "Average Ring C. (stdev) : "
        + str(metric_generic_generated.ring_counts_avg)
        + " ("
        + str(metric_generic_generated.ring_counts_std)
        + ")"
    )

    plot_dir = output_folder / "plot"
    os.makedirs(plot_dir, exist_ok=True)

    plot_distributions(
        [
            metric_generic_dataset.QED_scores.reshape(1, -1).squeeze(),
            metric_generic_generated.QED_scores.reshape(1, -1).squeeze(),
        ],
        plot_name="QED",
        output_folder=plot_dir,
    )

    plot_distributions(
        [
            metric_generic_dataset.logP_scores.reshape(1, -1).squeeze(),
            metric_generic_generated.logP_scores.reshape(1, -1).squeeze(),
        ],
        plot_name="logP",
        output_folder=plot_dir,
    )

    plot_distributions(
        [
            metric_generic_dataset.molecular_weights.reshape(1, -1).squeeze(),
            metric_generic_generated.molecular_weights.reshape(1, -1).squeeze(),
        ],
        plot_name="Molecular Weight",
        output_folder=plot_dir,
    )


if __name__ == "__main__":
    main()

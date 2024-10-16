# coding=utf-8
import sys
import os
import networkx as nx
from rdkit import Chem

from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from molvs import validate as mv

import numpy as np
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class GenericMetrics:
    QED_scores = list()
    logP_scores = list()
    molecular_weights = list()
    ring_counts = list()

    QED_scores_avg: float = 0
    QED_scores_std: float = 0
    logP_scores_avg: float = 0
    logP_scores_std: float = 0
    molecular_weights_avg: float = 0
    molecular_weights_std: float = 0
    ring_counts_avg: float = 0
    ring_counts_std: float = 0


def calculate_generic_metric(valid_molecules) -> GenericMetrics:

    progress_bar = tqdm(
        enumerate(valid_molecules),
        ascii="░▒▓",
        desc="Calculating Generic Metrics",
        total=len(valid_molecules),
    )
    metrics = GenericMetrics()

    for index, mol in progress_bar:
        try:
            # prepare molecule copy for score calculation
            Chem.SanitizeMol(mol)
            # calculate QED score with default rdkit parameters
            metrics.QED_scores.append(QED.qed(mol))
            # calculate logP score with default rdkit parameters
            metrics.logP_scores.append(Crippen.MolLogP(mol))
            # calculate molecular weight
            metrics.molecular_weights.append(Descriptors.MolWt(mol))
            # calculate number of rings
            metrics.ring_counts.append(rdMolDescriptors.CalcNumRings(mol))
        except:
            print("error on sanitize molecule", index)

    # cast descriptor lists to numpy array
    metrics.QED_scores = np.array(metrics.QED_scores, ndmin=2)
    metrics.logP_scores = np.array(metrics.logP_scores, ndmin=2)
    metrics.molecular_weights = np.array(metrics.molecular_weights, ndmin=2)
    metrics.ring_counts = np.array(metrics.ring_counts, ndmin=2)

    # calculate arithmetic mean and standard deviation of each descriptor over the generated graph set
    metrics.QED_scores_avg = np.mean(metrics.QED_scores)
    metrics.QED_scores_std = np.std(metrics.QED_scores)
    metrics.logP_scores_avg = np.mean(metrics.logP_scores)
    metrics.logP_scores_std = np.std(metrics.logP_scores)
    metrics.molecular_weights_avg = np.mean(metrics.molecular_weights)
    metrics.molecular_weights_std = np.std(metrics.molecular_weights)
    metrics.ring_counts_avg = np.mean(metrics.ring_counts)
    metrics.ring_counts_std = np.std(metrics.ring_counts)

    return metrics


# function that returns True if two nodes represent the same atom type
def node_equality(n1, n2):
    return n1["features"] == n2["features"]


# function that returns True if the two edges represent the same bond type
def edge_equality(e1, e2):
    return e1["features"] == e2["features"]


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
                error_list.append(
                    "Hydrogen atom "
                    + str(i)
                    + " has incorrect valence : "
                    + str(valence)
                )
        elif label == "C":
            if valence != 4:
                error_list.append(
                    "Carbon atom " + str(i) + " has incorrect valence : " + str(valence)
                )
        elif label == "N":
            if valence not in [3, 5]:
                error_list.append(
                    "Nitrogen atom "
                    + str(i)
                    + " has incorrect valence : "
                    + str(valence)
                )
        elif label == "O":
            if valence != 2:
                error_list.append(
                    "Oxygen atom " + str(i) + " has incorrect valence : " + str(valence)
                )
        elif label == "F":
            if valence != 1:
                error_list.append(
                    "Fluorine atom "
                    + str(i)
                    + " has incorrect valence : "
                    + str(valence)
                )
    # return the list of errors found in this molecular graph (if the list is empty the molecule is valid)
    return error_list


def check_Validity(graphs: list, molecules: Chem.RWMol, output_filename: str = None):
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
    if output_filename is not None and not os.path.exists(output_filename):
        out_file = open(output_filename, "w")
        out_file.close()
    progress_bar = tqdm(range(len(molecules)), ascii="░▒▓", desc="Checking validity")
    for i in progress_bar:

        validation_errors = Validate(graphs[i])
        if ValidateWithSanitization(molecules[i]) and not validation_errors:
            valid_indices.append(i)
            valid_molecules.append(molecules[i])
        elif output_filename is not None:
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

    progress_bar = tqdm(
        range(len(valid_indices)), ascii="░▒▓", desc="Checking Uniqueness"
    )
    for i in progress_bar:
        ii = valid_indices[i]
        unique = True
        for j in range(len(unique_graphs)):
            if nx.is_isomorphic(
                graphs[ii], unique_graphs[j], node_equality, edge_equality
            ):
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
            if nx.is_isomorphic(
                novel_graphs[j], training_set_graph, node_equality, edge_equality
            ):
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

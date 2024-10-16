# coding=utf-8
import sys
import os
import numpy as np
import networkx as nx

from rdkit import Chem
import argparse
from tqdm import tqdm
import pathlib


# Aggiungi la directory principale del progetto al PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ---
from utils.data_graphvae import load_QM9_metric
from utils.graph_utils import matrix_graph_to_nx, nx_to_mol
from .molecular_metric import check_Validity, calculate_generic_metric, GenericMetrics
from utils.utils import find_directory


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


def load_array_metrics_from_directory(directory: str, dataset_size: int):
    metric_array = {}
    metrics = GenericMetrics()
    # controllo se la cartella esiste
    if not pathlib.Path(directory).exists():
        print(f"Cartella {directory} non trovata")
        # mi calcolo le metriche
        dataset_directory = find_directory(directory, "QM9")
        metrics = calculate_dataset_metrics(data_path=dataset_directory, dataset_size=dataset_size)

    # controllo i sotto file della directory
    for file in os.listdir(directory):
        # carico gli arrey delle metriche
        if file.endswith(".txt"):
            # carico i dati
            try:
                setattr(
                    metrics,
                    file.replace(".txt", ""),
                    np.loadtxt(pathlib.Path(directory) / file, delimiter=","),
                )
            except:
                pass

    return metrics


def calculate_dataset_metrics(data_path: str, dataset_size: int):
    atom_label = {0: "C", 1: "N", 2: "O", 3: "F"}
    # bond_type = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}
    bond_type = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    output_folder = pathlib.Path(data_path, f"metrics_{dataset_size}", "stats")
    os.makedirs(output_folder, exist_ok=True)
    output_filename_summary = output_folder / "summary_metrics.txt"

    dataset_qm9 = load_QM9_metric(path=data_path, max_num_nodes=9)
    training_graphs_list = []
    for data in tqdm(dataset_qm9[0:dataset_size], desc="Loading training graphs"):
        graph = matrix_graph_to_nx(data.adj, data.x, data.edge_attr)
        training_graphs_list.append(graph)

    print("Dataset size: " + str(dataset_size))

    # translate each graph in a rdkit.Chem.RWMol object
    out_file = open(output_filename_summary, "w")
    out_file.write(f"Evaluation of metrics over generation run n")
    out_file.close()

    # translate each graph in a rdkit.Chem.RWMol object
    molecules = list()
    for i, G in enumerate(training_graphs_list):
        print("Translating graph " + str(i + 1) + " of " + str(len(training_graphs_list)), end="\r")
        mol = nx_to_mol(G, bond_type=bond_type, atom_label=atom_label)
        molecules.append(mol)
    print("")

    _, valid_molecules = check_Validity(training_graphs_list, molecules)

    # calculate chemical descriptors of each molecule
    metrics = calculate_generic_metric(valid_molecules)

    # save vectors of statistics to files for later use
    np.savetxt(output_folder / "QED_scores.txt", metrics.QED_scores, delimiter=" , ")
    np.savetxt(output_folder / "logP_scores.txt", metrics.logP_scores, delimiter=" , ")
    np.savetxt(output_folder / "molecular_weights.txt", metrics.molecular_weights, delimiter=" , ")
    np.savetxt(output_folder / "ring_counts.txt", metrics.ring_counts, delimiter=" , ")

    # print results to file
    print("Printing results to output file : ", str(output_folder))

    with open(output_filename_summary, "a") as f:
        f.write("Molecules in Set : " + str(len(training_graphs_list)) + "\n")
        f.write("Valid Molecules : " + str(len(valid_molecules)) + "\n")
        f.write(f"Average QED (stdev) : {str(metrics.QED_scores_avg)} ({str(metrics.QED_scores_std)})\n")
        f.write(f"Average logP (stdev) : {str(metrics.logP_scores_avg)} ({str(metrics.logP_scores_std)})\n")
        f.write(
            f"Average Mol.Wt. (stdev) : {str(metrics.molecular_weights_avg)} ({str(metrics.molecular_weights_std)})\n"
        )
        f.write(
            f"Average Ring C. (stdev) : {str(metrics.ring_counts_avg)} ({str(metrics.ring_counts_std)})\n"
        )

    return metrics


def main() -> None:
    prog_args = arg_parse()

    DEFAULT_dataset_size = 133582
    dataset_type = "QM9"

    dataset_size = (
        prog_args.num_samples
        if prog_args.num_samples is not None and prog_args.num_samples <= DEFAULT_dataset_size
        else DEFAULT_dataset_size
    )

    # setup path and dir
    script_path = os.path.dirname(os.path.realpath(__file__))
    repo_path = os.path.abspath(os.path.join(script_path, os.pardir))
    data_path = pathlib.Path(repo_path).parent / "data" / dataset_type

    os.makedirs(data_path, exist_ok=True)

    output_folder = pathlib.Path(data_path, f"metrics_{dataset_size}", "stats")

    metrics = load_array_metrics_from_directory(directory=output_folder, dataset_size=dataset_size)

    exit()

    # print results to screen
    print("Average QED (stdev) : " + str(metrics.QED_scores_avg) + " (" + str(metrics.QED_scores_std) + ")")
    print(
        "Average logP (stdev) : " + str(metrics.logP_scores_avg) + " (" + str(metrics.logP_scores_std) + ")"
    )
    print(
        "Average Mol.Wt. (stdev) : "
        + str(metrics.molecular_weights_avg)
        + " ("
        + str(metrics.molecular_weights_std)
        + ")"
    )
    print(
        "Average Ring C. (stdev) : "
        + str(metrics.ring_counts_avg)
        + " ("
        + str(metrics.ring_counts_std)
        + ")"
    )


if __name__ == "__main__":
    main()

import os
import torch
from tqdm import tqdm
import argparse
import csv
from datetime import datetime
import json

# ---
from GraphVAE.model_graphvae import GraphVAE
from script.LatentDiffusion.model_latent_old import SimpleUnet
from utils.data_graphvae import load_QM9
from utils import graph_to_mol, set_seed
from evaluate import calc_metrics
from train_diffusion import sample_timestep


def test(
    model_vae: GraphVAE,
    val_loader: torch.utils.data.DataLoader,
    latent_dimension: int,
    device,
    treshold_adj: float,
    treshold_diag: float,
    model_diffusion: SimpleUnet = None,
):
    model_vae.eval()
    if model_diffusion is not None:
        print("TEST ON LATENT DIFFUSION")
        model_diffusion.eval()
    else:
        print("TEST ON GRAPHVAE")
    smiles_pred = []
    smiles_true = []
    edges_medi_pred = 0
    edges_medi_true = 0
    val_pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        colour="red",
        desc="Batch",
        position=0,
        leave=False,
    )
    with torch.no_grad():
        for idx, data in val_pbar:
            z = torch.rand(len(data["smiles"]), latent_dimension).to(device)

            # print("-----")
            if model_diffusion is not None:

                z = sample_timestep(z, torch.tensor([0], device=device), model_diffusion).to(device)

            (recon_adj, recon_node, recon_edge, n_one) = model_vae.generate(z, treshold_adj, treshold_diag)
            for idx_data, elem in enumerate(data["smiles"]):
                if n_one[idx_data] == 0:
                    mol = None
                    smile = None
                else:
                    mol, smile = graph_to_mol(
                        recon_adj[idx_data].cpu(),
                        recon_node[idx_data],
                        recon_edge[idx_data].cpu(),
                        False,
                        True,
                    )
                # if smile == "" or smile == None:
                #     val_pbar.write("ERROR impossibile creare Molecola")

                smiles_pred.append(smile)
                smiles_true.append(elem)

                edges_medi_pred += n_one[idx_data]
                edges_medi_true += data["num_edges"][idx_data]

    unique_smiles_list, validity, uniqueness, novelty = calc_metrics(smiles_true, smiles_pred)
    edges_medi_pred = (edges_medi_pred / len(smiles_pred)).item()
    edges_medi_true = (edges_medi_true / len(smiles_true)).item()

    return (unique_smiles_list, validity, uniqueness, novelty, edges_medi_pred, edges_medi_true)


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument(
        "--treshold_adj",
        dest="treshold_adj",
        type=float,
        help="Treshold dei valori fuori dalla diagonale della matrice adiacente",
    )
    parser.add_argument(
        "--treshold_diag",
        dest="treshold_diag",
        type=float,
        help="Treshold dei valori nella diagonale della matrice adiacente",
    )

    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")

    parser.add_argument(
        "--max_num_nodes",
        dest="max_num_nodes",
        type=int,
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by training data.",
    )

    parser.add_argument("--num_examples", type=int, dest="num_examples", help="Number of examples")
    parser.add_argument("--latent_dimension", type=int, dest="latent_dimension", help="Latent Dimension")
    parser.add_argument("--epochs", type=int, dest="epochs", help="Number of epochs")
    parser.add_argument("--device", type=str, dest="device", help="cuda or cpu")
    parser.set_defaults(
        treshold_adj=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        treshold_diag=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        batch_size=1000,
        num_examples=20000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return parser.parse_args()


def write_csv(file_csv, header, risultati):
    # Controlla se il file esiste
    file_esiste = os.path.isfile(file_csv)

    if len(risultati) < len(header):
        risultati.append(datetime.now().strftime("%Y-%m-%d"))

    # Apertura del file in modalità append
    with open(file_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_esiste:
            # Scrivi l'header se il file non esiste
            writer.writerow(header)
        # Aggiungi i risultati
        writer.writerow(risultati)


def treshold_search(
    model_vae,
    model_diff,
    list_treshold_adj,
    list_treshold_diag,
    test_dataset_loader,
    latent_dimension,
    device,
    folder_base,
    test_samples: int = 1000,
):
    header = [
        "Treshold ADJ",
        "Treshold DIAG",
        "Validity %",
        "Uniqueness %",
        "Novelty %",
        "Edges Medi pred",
        "Edges Medi true",
        "Unique Smiles Generated List",
        "Data",
    ]
    print("------- TRESHOLD SEARCH -------")
    for t_adj in list_treshold_adj:
        for t_diag in list_treshold_diag:
            print("--- Testing Treshold Adj:", t_adj, "Treshold Diag:", t_diag)
            (
                unique_smiles_list,
                validity,
                uniqueness,
                novelty,
                edges_pred,
                edges_true,
            ) = test(
                model_vae,
                test_dataset_loader,
                latent_dimension,
                device,
                t_adj,
                t_diag,
                model_diff,
            )

            print(f"--- Validità: {validity:.2%}")
            print(f"--- Unicità: {uniqueness:.2%}")
            print(f"--- Novità: {novelty:.2%}")

            print("--- Numero edge medi predetti: ", edges_pred)
            print("--- Numero edge medi true: ", edges_true)

            print("------- ............ -------")

            results = [
                t_adj,
                t_diag,
                validity * 100,
                uniqueness * 100,
                novelty * 100,
                edges_pred,
                edges_true,
                unique_smiles_list,
            ]

            nome_file = f"test_result_{test_samples}_v2.csv"
            nome_file_path = os.path.join(folder_base, nome_file)
            write_csv(nome_file_path, header, results)


def load_GraphVAE(model_folder: str = "", device="cpu"):

    json_files = [file for file in os.listdir(model_folder) if file.endswith(".json")]
    # Cerca i file con estensione .pth
    pth_files = [file for file in os.listdir(model_folder) if file.endswith(".pth")]

    if json_files:
        hyper_file_json = os.path.join(model_folder, json_files[0])
    else:
        hyper_file_json = None

    if pth_files:
        model_file_pth = os.path.join(model_folder, pth_files[0])
    else:
        model_file_pth = None

    with open(hyper_file_json, "r") as file:
        dati = json.load(file)
        hyper_params = dati[0]

    model_GraphVAE = GraphVAE(
        hyper_params["latent_dimension"],
        max_num_nodes=hyper_params["max_num_nodes"],
        max_num_edges=hyper_params["max_num_edges"],
        num_nodes_features=hyper_params["num_nodes_features"],
        num_edges_features=hyper_params["num_edges_features"],
        device=device,
    )

    data_model = torch.load(model_file_pth, map_location="cpu")
    model_GraphVAE.load_state_dict(data_model)

    model_GraphVAE.to(device)

    return model_GraphVAE, hyper_params


def load_Diffusion(model_folder: str = "", device="cpu"):
    json_files = [file for file in os.listdir(model_folder) if file.endswith(".json")]
    # Cerca i file con estensione .pth
    pth_files = [file for file in os.listdir(model_folder) if file.endswith(".pth")]

    if json_files:
        hyper_file_json = os.path.join(model_folder, json_files[0])
    else:
        hyper_file_json = None

    if pth_files:
        model_file_pth = os.path.join(model_folder, pth_files[0])
    else:
        model_file_pth = None

    with open(hyper_file_json, "r") as file:
        dati = json.load(file)
        try:
            hyper_params = dati[0]
        except:
            hyper_params = dati

    model_diffusion = SimpleUnet(
        hyper_params["latent_dimension"],
        hyper_params["down_channel"],
        hyper_params["time_emb_dim"],
    )

    model_diffusion.load_state_dict(torch.load(model_file_pth, map_location="cpu"))
    model_diffusion.to(device)

    return model_diffusion


if __name__ == "__main__":
    print("~" * 20, "TESTING", "~" * 20)
    set_seed(42)

    args_parsed = arg_parse()

    # loading dataset
    # num_examples = args_parsed.num_examples
    batch_size = args_parsed.batch_size
    device = args_parsed.device

    folder_base = "models"
    graph_vae_num_samples = 1000
    diffusion_num_samples = 1000
    version = "v3"
    experiment_model_vae_name = f"logs_GraphVAE_{version}_{graph_vae_num_samples}"
    experiment_model_diffusion_name = (
        f"logs_Diffusion_{version}_{diffusion_num_samples}_from_{graph_vae_num_samples}"
    )

    model_folder_vae_base = os.path.join(folder_base, experiment_model_vae_name)
    model_folder_diff_base = os.path.join(folder_base, experiment_model_diffusion_name)

    model_vae, hyperparams = load_GraphVAE(model_folder=model_folder_vae_base, device=device)

    model_diffusion = load_Diffusion(model_folder_diff_base, device=device)
    # model_diffusion = None

    # setto il numero di esempi su cui fare il test tanti quanti sono gli elementi del training
    if model_diffusion is not None:
        num_examples = diffusion_num_samples
    else:
        num_examples = graph_vae_num_samples

    # LOAD DATASET QM9:
    (_, _, train_dataset_loader, _, val_dataset_loader, max_num_nodes) = load_QM9(
        hyperparams["max_num_nodes"],
        num_examples,
        batch_size,
        dataset_split_list=(0.7, 0.0, 0.3),
    )

    max_num_edges = hyperparams["max_num_edges"]

    print("----------------" * 2)
    print("Test set: {}".format(len(train_dataset_loader) * batch_size))
    print("max num edges:", max_num_edges)
    print("max num nodes:", max_num_nodes)
    print("num edges features", hyperparams["num_edges_features"])
    print("num nodes features", hyperparams["num_nodes_features"])

    print(model_folder_vae_base if model_diffusion is None else model_folder_diff_base)

    treshold_search(
        model_vae=model_vae,
        model_diff=model_diffusion,
        list_treshold_adj=args_parsed.treshold_adj,
        list_treshold_diag=args_parsed.treshold_diag,
        test_dataset_loader=train_dataset_loader,
        latent_dimension=hyperparams["latent_dimension"],
        device=device,
        folder_base=(model_folder_vae_base if model_diffusion is None else model_folder_diff_base),
        test_samples=num_examples,
    )

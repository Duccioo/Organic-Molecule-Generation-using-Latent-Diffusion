import os
import pathlib
import random
import re
from datetime import datetime
import pickle
from tqdm import tqdm
import argparse
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ---
from LatentDiffusion.model_latent import LatentDiffusionModel
from GraphVAE.model_graphvae import GraphVAE, load_graphvae_from_folder

from utils.data_graphvae import load_QM9_dataloader
from utils import (
    set_seed,
    matrix_graph_to_mol,
    save_checkpoint,
    count_parameters,
    logs_and_print,
    log_csv,
    check_base_dir,
)
from utils.summary import Summary
from utils.load_and_save import load_latest_checkpoint, save_checkpoint
from utils.plotting import plot_losses
from utils.graph_utils import matrix_graph_to_nx
from calculate_metrics import check_Validity, check_Uniqueness, save_to_smiles
from pca import plot_pca_distributions, pca_2component


# Funzione di perdita
def diffusion_loss(noise, predicted_noise):
    return nn.MSELoss()(noise, predicted_noise)


def validation(
    model: LatentDiffusionModel,
    val_loader,
    treshold_adj: float = 0.5,
    treshold_diag: float = 0.5,
    steps: int = 500,
    model_folder: str = "",
):
    device = next(model.parameters()).device
    model.eval()
    val_output_filename = pathlib.Path(model_folder, "val_output.txt")
    unique_molecules_filename = pathlib.Path(model_folder, "unique_molecules.txt")
    generated_molecules = []
    generated_graphs = []
    val_info = {
        "validity": 0.0,
        "uniqueness": 0.0,
        "novelty": 0.0,
        "unique_overtotal": 0.0,
        "novel_overtotal": 0.0,
    }

    step = 0
    step_size_mean = 0  # Passo con cui si muove la griglia ad ogni iterazione
    step_size_std = 50  # Passo con cui si muove la griglia ad ogni iterazione
    initial_mean = torch.zeros(model.latent_dim).to(device)  # Media iniziale
    initial_std = torch.ones(model.latent_dim).to(
        device
    )  # Deviazione standard iniziale
    graph_not_generated = 0
    p_bar_val = tqdm(
        val_loader,
        position=0,
        leave=False,
        total=len(val_loader),
        colour="yellow",
        desc="Validate",
    )
    with torch.no_grad():

        for batch_val in p_bar_val:

            # Calcola la nuova media spostandola di uno step_size ad ogni iterazione
            new_mean = initial_mean + step * step_size_mean
            # Fattore casuale per cambiare il segno della media con uguale probabilità
            sign_flip = torch.randint(0, 2, (model.latent_dim,)).to(device) * 2 - 1
            new_mean = new_mean * sign_flip  # Applica il cambio di segno casuale

            # Mantieni una deviazione standard costante o modificala se necessario
            new_std = initial_std + step * step_size_std
            sign_flip = torch.randint(0, 2, (model.latent_dim,)).to(device) * 2 - 1
            new_std = new_std * sign_flip  # Applica il cambio di segno casuale

            z = (
                new_std
                * torch.rand(len(batch_val["smiles"]), model.latent_dim).to(device)
                + new_mean
            )
            adj, features_nodes, features_edges, _ = model.decode(
                z, steps, treshold_adj, treshold_diag
            )

            while torch.isnan(adj).any():
                graph_not_generated += 1

                z = (
                    new_std
                    * torch.rand(len(batch_val["smiles"]), model.latent_dim).to(device)
                    + new_mean
                )
                adj, features_nodes, features_edges, _ = model.decode(
                    z, steps, treshold_adj, treshold_diag
                )

                if graph_not_generated > 20:
                    print("forse ci sono dei problemi....... No graph generated")
                    break

            if graph_not_generated > 20:
                print("forse ci sono dei problemi....... No graph generated")
                break

            step += 1
            for index_val, elem in enumerate(batch_val["smiles"]):

                # translate each graph in a rdkit.Chem.RWMol object
                graph, molecule = matrix_graph_to_mol(
                    adj[index_val].cpu(),
                    features_nodes[index_val].cpu(),
                    features_edges[index_val].cpu(),
                )
                generated_molecules.append(molecule)
                generated_graphs.append(graph)

        valid_indices, valid_molecules = check_Validity(
            generated_graphs, generated_molecules, val_output_filename
        )
        if len(valid_molecules) == 0:
            val_info["uniqueness"] = 0
            val_info["unique_overtotal"] = 0
        else:
            unique_graphs, unique_molecules, _ = check_Uniqueness(
                generated_graphs, valid_molecules, valid_indices
            )
            val_info["uniqueness"] = float(len(unique_molecules) / len(valid_molecules))
            val_info["unique_overtotal"] = float(
                len(unique_molecules) / len(generated_graphs)
            )
            save_to_smiles(unique_molecules, unique_molecules_filename)
        if len(generated_graphs) != 0:
            val_info["validity"] = float(len(valid_molecules) / len(generated_graphs))
        else:
            val_info["validity"] = 0
    return val_info["validity"], val_info


def train_one_epoch(
    model_ldm: LatentDiffusionModel, model_vae: GraphVAE, train_loader, optimizer
):
    model_ldm.train()
    device = next(model_ldm.parameters()).device

    max_size_atom = 9
    running_loss = 0.0
    train_epoch_loss = 0.0
    training_info = {
        "datetime": [],
        "total_loss": [],
    }
    # BATCH FOR LOOP
    p_bar_batch = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        position=0,
        leave=False,
        colour="CYAN",
    )
    for i, data in p_bar_batch:
        optimizer.zero_grad()

        features_nodes = data.x.float().to(device)
        features_edges = data.edge_attr.float().to(device)
        edge_index = data.edge_index.to(device)
        features_edge_padded = data.edge_attr_removed_duplicates.to(device)

        base_sequence = torch.arange(data.x.shape[0] // max_size_atom)
        data.batch = base_sequence.repeat_interleave(max_size_atom).to(device)

        # Codifica i dati nello spazio latente
        with torch.no_grad():
            z, vae_mu, vae_var = model_vae.encoder(
                features_nodes, features_edges, edge_index, data.batch
            )

        # Genera un tempo casuale
        t = torch.randint(0, model_ldm.num_timesteps, (z.shape[0],)).to(device)

        # Aggiungi rumore ai dati codificati
        noisy_z, noise = model_ldm.add_noise_2(z, t)

        # Predici il rumore
        # print(noisy_z.shape)
        # print(t.shape)
        # exit()
        predicted_noise = model_ldm(noisy_z, t)

        # Calcola la perdita
        loss = diffusion_loss(noise, predicted_noise)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        training_info["datetime"].append(datetime.now())
        training_info["total_loss"].append(loss.item())
        p_bar_batch.set_description(f"Loss: {loss.item():.4f}")

    p_bar_batch.close()

    train_epoch_loss = running_loss / (len(train_loader))

    return train_epoch_loss, training_info


def train(
    model_ldm,
    model_vae,
    train_loader,
    val_loader,
    summary: Summary,
    epochs: int = 50,
    learning_rate: float = 0.001,
    scheduler_patience: int = 3,
):

    optimizer = Adam(model_ldm.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=scheduler_patience, factor=0.5
    )

    logs_csv_filename = pathlib.Path(summary.directory_log, "metric_training.csv")
    logs_plot_filename = pathlib.Path(summary.directory_img, "plot_training.png")

    epochs_saved = 0
    val_treshold_adj = 0.5
    val_treshold_diag = 0.3
    train_epochs_info = []
    validation_saved = []
    validity_epoch = 0.0, 0.0

    # Checkpoint load
    epochs_saved, data_saved = load_latest_checkpoint(
        summary.directory_checkpoint, model_ldm, optimizer, scheduler
    )

    if epochs_saved is not None:
        epochs_saved = data_saved["epoch"]
        train_epochs_info = data_saved["loss"]
        validation_saved = data_saved["other"]
        print(f"start from checkpoint at epoch {epochs_saved}")
    else:
        logs_and_print(summary.directory_log, "Start training", recrate=True)
        epochs_saved = 0

    logs_and_print(
        summary.directory_log,
        f"Learnable Model parameters: {count_parameters(model_ldm)[0]}",
        print_message=False,
    )
    logs_and_print(
        summary.directory_log,
        f"Not Learnable Model parameters: {count_parameters(model_ldm)[1]}",
        print_message=False,
    )

    logs_and_print(summary.directory_log, dot_line=True)

    p_bar_epoch = tqdm(
        range(1, epochs + 1),
        desc="epochs",
        position=1,
        colour="blue",
        leave=True,
        smoothing=0.05,
    )
    for epoch in p_bar_epoch:
        if epoch <= epochs_saved:
            continue

        model_ldm.train()
        train_epoch_loss, training_info = train_one_epoch(
            model_ldm, model_vae, train_loader, optimizer
        )
        train_epochs_info.append(training_info)

        text_epoch = f"Epoch {epoch} - Loss: {train_epoch_loss:.4f}"

        if epoch % 5 == 0 and epoch != 0:
            # Validation each 5 epoch:
            validity_epoch, validation_info = validation(
                model_ldm,
                val_loader,
                val_treshold_adj,
                val_treshold_diag,
            )

            validation_saved.append(validation_info)
            p_bar_epoch.write("Saving checkpoint...")
            save_checkpoint(
                model_ldm,
                optimizer,
                epoch,
                scheduler,
                loss=train_epochs_info,
                other=validation_saved,
                directory=summary.directory_checkpoint,
            )
            text_epoch = f"Epoch {epoch} - Loss: {train_epoch_loss:.4f}, Validity: {100*validity_epoch:.2f}% Uniqueness: {100*validation_saved[-1]['uniqueness']:.2f}%"

        logs_and_print(summary.directory_log, text_epoch, print_message=True)
        logs_and_print(summary.directory_log, dot_line=True)

    train_loss = [
        valore
        for dizionario in train_epochs_info
        for valore in dizionario["total_loss"]
    ]
    train_dict_losses = {"train_total_loss": train_loss}
    val_dict = {
        "Validity": [elm["validity"] for elm in validation_saved],
        "Uniqueness": [elm["uniqueness"] for elm in validation_saved],
    }

    plot_losses(
        train_dict_losses,
        val_dict,
        num_batches=len(train_loader),
        output_file=logs_plot_filename,
    )
    log_csv(logs_csv_filename, train_dict_losses)

    return model_ldm


def generate_and_save_graphs(
    model: LatentDiffusionModel,
    N: int,
    output_path: str = "generated_graphs",
    steps: int = 500,
    treshold_adj: float = 0.5,
    treshold_diag: float = 0.5,
):
    # Creazione della cartella per i grafi generati
    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)
    device = next(model.parameters()).device

    # Generazione del grafo dal vettore di rumore
    factor = int(N * 0.05)
    progress_bar = tqdm(range(N // factor), colour="green", desc="Generating graphs")
    graph_list = []
    graphs_not_generated = 0

    step_size_mean = 0.0  # Passo con cui si muove la griglia ad ogni iterazione
    step_size_std = 50  # Passo con cui si muove la griglia ad ogni iterazione
    initial_mean = torch.zeros(model.latent_dim).to(device)  # Media iniziale
    initial_std = torch.ones(model.latent_dim).to(
        device
    )  # Deviazione standard iniziale
    step = 0

    for i in progress_bar:
        # Calcola la nuova media spostandola di uno step_size ad ogni iterazione
        new_mean = initial_mean + step * step_size_mean
        # Fattore casuale per cambiare il segno della media con uguale probabilità
        sign_flip = torch.randint(0, 2, (model.latent_dim,)).to(device) * 2 - 1
        new_mean = new_mean * sign_flip  # Applica il cambio di segno casuale

        # Mantieni una deviazione standard costante o modificala se necessario
        new_std = initial_std + step * step_size_std
        sign_flip = torch.randint(0, 2, (model.latent_dim,)).to(device) * 2 - 1
        new_std = new_std * sign_flip

        # Applica il "reparameterization trick"
        z = new_std * torch.randn(factor, model.latent_dim).to(device) + new_mean

        adj, features_nodes, features_edges, _ = model.decode(
            z, steps, treshold_adj, treshold_diag
        )

        while torch.isnan(adj).any():
            graphs_not_generated += 1
            z = new_std * torch.randn(factor, model.latent_dim).to(device) + new_mean
            adj, features_nodes, features_edges, _ = model.decode(
                z, steps, treshold_adj, treshold_diag
            )

        step += 1

        for j in range(factor):
            if torch.isnan(adj[j]).any():
                graphs_not_generated += 1
                continue

            # Creazione del grafo e salvataggio
            graph_i = matrix_graph_to_nx(
                adj[j], features_nodes[j], features_edges[j], filename=None
            )
            graph_list.append(graph_i)

        progress_bar.set_description(f"Generated graphs G_{i*factor}-{(i+1)*factor}")

    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)

    filename_path = pathlib.Path(output_path, "graphs_generated.pkl")
    with open(filename_path, "wb") as f:
        pickle.dump(graph_list, f, pickle.HIGHEST_PROTOCOL)

    print(f"Graphs not generated: {graphs_not_generated}")


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument(
        "--gen_path",
        type=str,
        dest="generated_graph_path",
        help="path to generated graphs",
    )
    parser.add_argument(
        "--model_graphvae_folder", type=str, help="path to model folder"
    )
    parser.add_argument("--graphvae_samples", type=str)
    parser.add_argument("--version", type=str)

    parser.set_defaults(
        generated_graph_path="generated_graphs_v5_approx",
        model_graphvae_folder="GraphVAE_v20.5_fingerprint_fs128_100000",
        graphvae_samples=100000,
        version="20.05_XX",
        changelog="Versione 10.10p del diffusion \nVersione 20.5 del graphvae\n,"
        + "\n",
    )
    return parser.parse_args()


def take_fs(stringa):
    pattern = r"fs(\d+)"
    match = re.search(pattern, stringa)
    if match:
        return int(match.group(1))
    else:
        return None


def main():

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prog_args = arg_parse()

    # setup path and dir
    script_path = os.path.dirname(os.path.realpath(__file__))
    repo_path = os.path.abspath(os.path.join(script_path, os.pardir))
    data_path = check_base_dir(repo_path, "data", "QM9")

    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EXAMPLES = 100000
    MAX_TIMESTEPS = 500
    epochs = 50
    learning_rate = 0.00001
    train_percentage = 0.7
    test_percentage = 0.0
    val_percentage = 0.3
    unet_depths = [80, 160, 320]
    time_emb_dim = 100
    experiment_model_type = "Diffusion"
    model_folder = "models"
    graph_vae_samples = prog_args.model_graphvae_folder.split("_")[-1]
    version = prog_args.version

    graph_vae_folder_name = prog_args.model_graphvae_folder
    experiment_folder = os.path.join(
        repo_path,
        model_folder,
        f"Diffusion_v{version}_"
        + str(NUM_EXAMPLES)
        + "_from_"
        + str(graph_vae_samples),
    )

    folder_GraphVAE = os.path.join(repo_path, model_folder, graph_vae_folder_name)
    print(folder_GraphVAE)

    model_vae, hyperparams = load_graphvae_from_folder(model_folder=folder_GraphVAE)
    model_vae.freeze_all_parameters()
    model_vae.eval()

    model_ldm = LatentDiffusionModel(
        model_vae,
        latent_dim=hyperparams["latent_dimension"],
        unet_in_channels=hyperparams["latent_dimension"],
        time_dim=time_emb_dim,
        num_timesteps=MAX_TIMESTEPS,
        unet_depths=unet_depths,
    )
    print("Learnable Model VAE parameters: ", count_parameters(model_vae)[0])
    print("Learnable Model LDM parameters: ", count_parameters(model_ldm)[0])

    model_ldm.to(device)
    model_vae.to(device)

    # LOAD DATASET QM9:
    (_, dataset_pad, train_dataset_loader, _, val_dataset_loader, _) = (
        load_QM9_dataloader(
            data_path,
            hyperparams["max_num_nodes"],
            NUM_EXAMPLES,
            BATCH_SIZE,
            validation_batch_multiplier=3,
            dataset_split_list=(train_percentage, test_percentage, val_percentage),
        )
    )

    training_effective_size = (
        len(train_dataset_loader) * train_dataset_loader.batch_size
    )
    validation_effective_size = len(val_dataset_loader) * val_dataset_loader.batch_size
    num_nodes_features = dataset_pad[0]["features_nodes"].shape[1]
    num_edges_features = dataset_pad[0]["features_edges"].shape[1]

    # prima di iniziare il training creo la cartella in cui salvere i checkpoints, modello, iperparametri, etc...
    model__hyper_params = []
    dataset__hyper_params = []
    dataset__hyper_params.append(
        {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "num_examples": NUM_EXAMPLES,
            "batch_size": BATCH_SIZE,
            "max_timesteps": MAX_TIMESTEPS,
            "training_percentage": train_percentage,
            "test_percentage": test_percentage,
            "val_percentage": val_percentage,
            "number_val_examples": validation_effective_size,
            "number_train_examples": training_effective_size,
            "graph_vae_model_name": graph_vae_folder_name,
        }
    )
    model__hyper_params.append(
        {
            "num_nodes_features": num_nodes_features,
            "num_edges_features": num_edges_features,
            "max_num_nodes": hyperparams["max_num_nodes"],
            "max_num_edges": hyperparams["max_num_edges"],
            "latent_dimension": hyperparams["latent_dimension"],
        }
    )

    summary = Summary(experiment_folder, experiment_model_type)
    summary.save_json(model__hyper_params, file_name="hyperparams.json")
    summary.save_json(dataset__hyper_params, file_name="dataset_hyperparams.json")
    summary.save_summary_training(
        dataset__hyper_params, hyperparams, random.choice(dataset_pad)
    )
    summary.changelog(prog_args.version, prog_args.changelog)

    # -- TRAINING -- :
    print("\nStart training...")
    trained_model_ldm = train(
        model_ldm,
        model_vae,
        train_dataset_loader,
        val_dataset_loader,
        summary,
        epochs,
        learning_rate,
    )

    # salvo il modello finale:
    final_model_name = "trained_Diffusion_FINAL.pth"
    final_model_path = os.path.join(summary.directory_base, final_model_name)
    torch.save(model_ldm.state_dict(), final_model_path)

    # ---- INFERENCE ----
    folder_generated_graphs = pathlib.Path(summary.directory_base, "generated_graphs")

    print("INFERENCE!")
    with torch.no_grad():
        model_ldm.eval()

        generate_and_save_graphs(
            model_ldm,
            NUM_EXAMPLES,
            output_path=folder_generated_graphs,
            steps=MAX_TIMESTEPS,
            treshold_adj=0.5,
            treshold_diag=0.3,
        )


if __name__ == "__main__":
    main()

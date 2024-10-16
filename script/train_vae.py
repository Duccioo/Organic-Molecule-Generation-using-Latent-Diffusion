import argparse
import pathlib
from datetime import datetime
import random
import pickle
import re

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import NeuralFingerprint
from tqdm import tqdm
from pca import pca_2component, plot_pca_distributions

# ---
from GraphVAE.model_graphvae import GraphVAE
from GraphVAE.losses import GraphVAE_loss, fingerprint_loss, approx_loss
import NeuralFingerprint
from utils.data_graphvae import load_QM9_dataloader
from utils import (
    save_checkpoint,
    set_seed,
    matrix_graph_to_mol,
    matrix_graph_to_nx,
    count_parameters,
    logs_and_print,
    log_csv,
    log_dict_to_json,
)
from utils.load_and_save import load_latest_checkpoint, save_checkpoint
from utils.plotting import plot_losses
from utils.summary import Summary
from calculate_metrics import check_Validity, check_Uniqueness, save_to_smiles


def validation(
    model: GraphVAE,
    val_loader,
    treshold_adj: float = 0.5,
    treshold_diag: float = 0.5,
    model_folder: str = "",
):
    device = next(model.parameters()).device
    model.eval()
    val_output_filename = pathlib.Path(model_folder, "val_output.txt")
    unique_molecules_filename = pathlib.Path(model_folder, "unique_molecules.txt")
    val_info = {
        "validity": 0.0,
        "uniqueness": 0.0,
        "novelty": 0.0,
        "unique_overtotal": 0.0,
        "novel_overtotal": 0.0,
    }
    generated_molecules = []
    generated_graphs = []

    step: int = 0
    step_size_mean: float = 0  # Passo con cui si muove la griglia ad ogni iterazione
    step_size_std: float = 50  # Passo con cui si muove la griglia ad ogni iterazione
    initial_mean = torch.zeros(model.latent_dim).to(device)  # Media iniziale
    initial_std = torch.ones(model.latent_dim).to(device)  # Deviazione standard iniziale

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

            # Applica il "reparameterization trick"
            z = new_std * torch.randn(len(batch_val["smiles"]), model.latent_dim).to(device) + new_mean
            adj, features_nodes, features_edges, _ = model.generate(z, treshold_adj, treshold_diag)

            while torch.isnan(adj).any():
                z = new_std * torch.randn(len(batch_val["smiles"]), model.latent_dim).to(device) + new_mean
                adj, features_nodes, features_edges, _ = model.generate(z, treshold_adj, treshold_diag)

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
            val_info["unique_overtotal"] = float(len(unique_molecules) / len(generated_graphs))
            save_to_smiles(unique_molecules, unique_molecules_filename)

        val_info["validity"] = float(len(valid_molecules)) / len(generated_graphs)

    return val_info["validity"], val_info


def train_one_epoch(
    model: GraphVAE, train_loader, model_fingerprint: NeuralFingerprint, optimizer, loss: GraphVAE_loss
):
    model.train()
    device = next(model.parameters()).device
    normalize_loss = False  # Se True, la loss viene normalizzata
    max_size_atom = 9
    running_loss = 0.0
    train_epoch_loss = 0.0
    training_info = {
        "datetime": [],
        "total_loss": [],
        "loss_recon": [],
        "loss_kl": [],
        "loss_adj": [],
        "loss_node": [],
        "loss_edge": [],
        "mean": [],
        "variance": [],
        "adj_recon": [],
        "m_mean": [],
        "m_std": [],
        "example_smiles": [],
        "example_adj": [],
        "example_node": [],
        "example_edge": [],
        "recon_adj_vec": [],
        "recon_node": [],
        "recon_edge": [],
    }

    # BATCH FOR LOOP
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        position=0,
        leave=False,
        colour="CYAN",
    )
    for i, data in progress_bar:
        optimizer.zero_grad()

        features_nodes = data.x.float().to(device)
        features_edges = data.edge_attr.float().to(device)
        adj_input = data.adj.float().to(device)
        edge_index = data.edge_index.to(device)
        features_edge_padded = data.edge_attr_removed_duplicates.to(device)

        base_sequence = torch.arange(data.x.shape[0] // max_size_atom)
        data.batch = base_sequence.repeat_interleave(max_size_atom).to(device)

        adj_vec, mu, var, node_recon, edge_recon = model(
            features_nodes, features_edges, edge_index, data.batch
        )

        global_loss, dict_losses = loss.calc_loss(
            adj_input,
            adj_vec,
            features_nodes,
            node_recon,
            features_edge_padded,
            edge_recon,
            data.batch,
            mu,
            var,
            fingerprint_model=model_fingerprint,
            normalize=normalize_loss,
        )

        global_loss.backward()
        optimizer.step()

        running_loss += global_loss.item()

        training_info["datetime"].append(datetime.now())
        training_info["total_loss"].append(dict_losses["total_loss"].item())
        training_info["loss_recon"].append(dict_losses["loss_recon"].item())
        training_info["loss_kl"].append(dict_losses["loss_kl"].item())
        training_info["loss_adj"].append(dict_losses["loss_adj"].item())
        training_info["loss_node"].append(dict_losses["loss_node"].item())
        training_info["loss_edge"].append(dict_losses["loss_edge"].item())
        # training_info["mean"].append(mu.cpu().detach().numpy())
        # training_info["variance"].append(var.cpu().detach().numpy())
        # training_info["adj_recon"].append(adj_vec.cpu().detach().numpy())
        training_info["m_mean"].append(torch.mean(mu, dim=0).cpu().detach().numpy())
        training_info["m_std"].append(torch.mean(var, dim=0).cpu().detach().numpy())
        progress_bar.set_description(f"Loss: {global_loss.item():.4f}")

    progress_bar.close()
    # salvo anche degli esempi di grafi reali e di grafi ricostruiti
    example_graph = train_loader.dataset[-1]
    training_info["example_smiles"].append(example_graph.smiles)
    training_info["example_adj"].append(example_graph.adj.cpu().detach().numpy().tolist())
    training_info["example_node"].append(example_graph.x.cpu().detach().numpy().tolist())
    training_info["example_edge"].append(example_graph.edge_attr.cpu().detach().numpy().tolist())
    training_info["recon_adj_vec"].append(adj_vec[0].cpu().detach().numpy().tolist())
    training_info["recon_node"].append(node_recon[0].cpu().detach().numpy().tolist())
    training_info["recon_edge"].append(edge_recon[0].cpu().detach().numpy().tolist())

    train_epoch_loss = running_loss / (len(train_loader))

    return train_epoch_loss, training_info


def train(
    model,
    train_loader,
    val_loader,
    summary: Summary,
    model_fingerprint: NeuralFingerprint,
    loss: GraphVAE_loss,
    epochs: int = 50,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 100,
    scheduler_patience: int = 3,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=scheduler_patience, factor=0.5)

    logs_csv_filename = pathlib.Path(summary.directory_log, "metric_training.csv")
    logs_plot_filename = pathlib.Path(summary.directory_img, "plot_training.png")
    logs_json_example_filename = pathlib.Path(summary.directory_log, "example.json")

    epochs_saved = 0
    val_treshold_adj = 0.5
    val_treshold_diag = 0.25
    train_epochs_info = []
    train_total_list = []
    validation_saved = []
    best_val = 0.00
    patience = early_stopping_patience

    # Checkpoint load
    epochs_saved, data_saved = load_latest_checkpoint(
        summary.directory_checkpoint, model, optimizer, scheduler
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
        f"Learnable Model parameters: {count_parameters(model)[0]}",
        print_message=False,
    )
    logs_and_print(
        summary.directory_log,
        f"Not Learnable Model parameters: {count_parameters(model)[1]}",
        print_message=False,
    )

    p_bar_epoch = tqdm(range(1, epochs + 1), desc="epochs", position=1, colour="blue", leave=True)

    for epoch in p_bar_epoch:
        if epoch <= epochs_saved:
            continue

        model.train()
        epoch_loss, batch_info = train_one_epoch(model, train_loader, model_fingerprint, optimizer, loss)
        val_validity, val_info = validation(
            model, val_loader, val_treshold_adj, val_treshold_diag, summary.directory_base
        )
        scheduler.step(val_validity)

        train_epochs_info.append(batch_info)
        train_total_list.append(epoch_loss)
        validation_saved.append(val_info)

        # print(f"\033[F\033[K", end="")  # Cancella l'ultima riga
        text_epoch = f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Validity: {100*val_validity:.2f}%, Uniqueness: {100*val_info['uniqueness']:.2f}%"

        logs_and_print(summary.directory_log, text_epoch, print_message=True)
        logs_and_print(summary.directory_log, dot_line=True)
        # p_bar_epoch.write(text_epoch)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            scheduler,
            loss=train_epochs_info,
            other=validation_saved,
            directory=summary.directory_checkpoint,
        )
        if val_validity > best_val:
            best_val = val_validity
            patience = early_stopping_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping at epoch", epoch)
                break

    train_date = [valore for dizionario in train_epochs_info for valore in dizionario["datetime"]]
    train_total_loss = [valore for dizionario in train_epochs_info for valore in dizionario["total_loss"]]
    train_recon_loss = [valore for dizionario in train_epochs_info for valore in dizionario["loss_recon"]]
    train_kl_loss = [valore for dizionario in train_epochs_info for valore in dizionario["loss_kl"]]
    train_edge_loss = [valore for dizionario in train_epochs_info for valore in dizionario["loss_edge"]]
    train_node_loss = [valore for dizionario in train_epochs_info for valore in dizionario["loss_node"]]
    train_adj_loss = [valore for dizionario in train_epochs_info for valore in dizionario["loss_adj"]]

    mean_mean_list = [valore for dizionario in train_epochs_info for valore in dizionario["m_mean"]]
    mean_std_list = [valore for dizionario in train_epochs_info for valore in dizionario["m_std"]]
    example_adj_list = [elm["example_adj"] for elm in train_epochs_info]
    example_node_list = [elm["example_node"] for elm in train_epochs_info]
    example_edge_list = [elm["example_edge"] for elm in train_epochs_info]
    example_smiles_list = [elm["example_smiles"] for elm in train_epochs_info]
    recon_adj_vec_list = [elm["recon_adj_vec"] for elm in train_epochs_info]
    recon_node_list = [elm["recon_node"] for elm in train_epochs_info]
    recon_edge_list = [elm["recon_edge"] for elm in train_epochs_info]

    assert (
        len(train_date) == len(train_total_loss) == len(train_recon_loss)
    ), "Error length of lists not equal"

    train_dict_losses = {
        "training_total_loss": train_total_loss,
        "training_recon_loss": train_recon_loss,
        "training_kl_loss": train_kl_loss,
        "training_edge_loss": train_edge_loss,
        "training_node_loss": train_node_loss,
        "training_adj_loss": train_adj_loss,
    }

    train_dict_info = {
        "datetime": train_date,
        "m_mean": mean_mean_list,
        "m_std": mean_std_list,
    }

    train_dict_example_and_recon = {
        "example_adj": example_adj_list,
        "example_node": example_node_list,
        "example_edge": example_edge_list,
        "example_smiles": example_smiles_list,
        "recon_adj_vec": recon_adj_vec_list,
        "recon_node": recon_node_list,
        "recon_edge": recon_edge_list,
    }

    val_dict = {
        "Validity": [elm["validity"] for elm in validation_saved],
        "Uniqueness": [elm["uniqueness"] for elm in validation_saved],
    }

    plot_losses(
        train_dict_losses,
        val_dict,
        num_batches=len(train_loader),
        output_file=logs_plot_filename,
        y_lim=torch.mean(torch.tensor(train_dict_losses["training_total_loss"])) * 1.5,
    )

    log_csv(logs_csv_filename, train_dict_info, train_dict_losses)
    log_dict_to_json(train_dict_example_and_recon, logs_json_example_filename)

    m_mean_1, m_mean_2 = pca_2component(train_dict_info["m_mean"])
    m_std_1, m_std_2 = pca_2component(train_dict_info["m_std"])
    plot_pca_distributions(
        directory=summary.directory_img,
        mean_data=(m_mean_1, m_mean_2),
        var_data=(m_std_1, m_std_2),
        filename="pca_2.png",
    )
    return model


def generate_and_save_graphs(
    model: GraphVAE,
    N: int,
    output_path: str = "generated_graphs",
    treshold_adj: float = 0.5,
    treshold_diag: float = 0.3,
):
    # Creazione della cartella per i grafi generati
    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)
    device = next(model.parameters()).device

    # Generazione del grafo dal vettore di rumore
    factor = int(N * 0.05)
    progress_bar = tqdm(range(N // factor), colour="green", desc="Generating graphs")
    graph_list = []
    graphs_not_generated = 0

    step_size = 0  # Passo con cui si muove la griglia ad ogni iterazione
    step_std_size = 50
    initial_mean = torch.zeros(model.latent_dim).to(device)  # Media iniziale
    initial_std = torch.ones(model.latent_dim).to(device)  # Deviazione standard iniziale
    step = 0

    for i in progress_bar:

        # Calcola la nuova media spostandola di uno step_size ad ogni iterazione
        new_mean = initial_mean + step * step_size
        # Fattore casuale per cambiare il segno della media con uguale probabilità
        sign_flip = torch.randint(0, 2, (model.latent_dim,)).to(device) * 2 - 1
        new_mean = new_mean * sign_flip  # Applica il cambio di segno casuale

        # Mantieni una deviazione standard costante o modificala se necessario
        new_std = initial_std + step * step_std_size
        sign_flip = torch.randint(0, 2, (model.latent_dim,)).to(device) * 2 - 1
        new_std = new_std * sign_flip

        # Applica il "reparameterization trick"
        z = new_std * torch.randn(factor, model.latent_dim).to(device) + new_mean
        adj, features_nodes, features_edges, _ = model.generate(z, treshold_adj, treshold_diag)

        while torch.isnan(adj).any():
            graphs_not_generated += 1
            z = new_std * torch.randn(factor, model.latent_dim).to(device) + new_mean
            adj, features_nodes, features_edges, _ = model.generate(z, treshold_adj, treshold_diag)

        step += 1

        for j in range(factor):
            if torch.isnan(adj[j]).any():
                graphs_not_generated += 1
                continue

            # Creazione del grafo e salvataggio
            graph_i = matrix_graph_to_nx(adj[j], features_nodes[j], features_edges[j], filename=None)
            graph_list.append(graph_i)

        progress_bar.set_description(f"Generated graphs G_{i*factor}-{(i+1)*factor}")

    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)

    filename_path = pathlib.Path(output_path, "graphs_generated.pkl")
    with open(filename_path, "wb") as f:
        pickle.dump(graph_list, f, pickle.HIGHEST_PROTOCOL)

    print(f"Graphs not generated: {graphs_not_generated}")


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data",
    )
    parser.add_argument(
        "--max_num_nodes",
        dest="max_num_nodes",
        type=int,
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by training data.",
    )

    parser.add_argument("--num_examples", type=int, dest="num_examples", help="Number of examples")
    parser.add_argument("--latent_dimension", type=int, dest="latent_dimension", help="Latent Dimension")
    parser.add_argument("--epochs", type=int, dest="epochs", help="Number of epochs")
    parser.add_argument(
        "--train_percentage",
        type=int,
        dest="train_dataset_percentage",
        help="Train dataset percentage",
    )
    parser.add_argument(
        "--test_percentage",
        type=int,
        dest="test_dataset_percentage",
        help="Train dataset percentage",
    )
    parser.add_argument(
        "--val_percentage",
        type=int,
        dest="val_dataset_percentage",
        help="Train dataset percentage",
    )
    parser.add_argument("--device", type=str, help="cuda or cpu")
    parser.add_argument("--loss__fingerprint_model_name", type=str)

    parser.set_defaults(
        model__latent_dimension=80,
        model__num_layers_encoder=5,
        model__num_layers_decoder=3,
        model__growth_factor_encoder=2,
        model__growth_factor_decoder=1,
        # ---
        training__lr=0.001,
        training__epochs=30,
        # ---
        dataset__batch_size=32,
        dataset__num_workers=1,
        dataset__max_num_nodes=9,
        dataset__num_examples=100000,
        dataset__train_percentage=0.7,
        dataset__test_percentage=0.0,
        dataset__val_percentage=0.3,
        # ---
        loss__type="approx_loss",
        loss__fingerprint_model_name="nn_fingerprint_qm9_fs128_v2",
        # ---
        loss__recon_w=2.0,  # 2.0
        loss__kl_w=1.50,
        loss__edge_w=1,
        loss__node_w=1.0,
        loss__adj_w=0.10,
        # ---
        version="20.4",
        changelog="Dal modello 20.0 però con la loss approssimata (dal paper famoso...)\n",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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

    prog_args = arg_parse()
    device = prog_args.device

    experiment_model_type = "GRAPH VAE"
    model_folder = "models"
    fingerprint_size = take_fs(prog_args.loss__fingerprint_model_name)
    version = f"{prog_args.version}_{prog_args.loss__type}_fs{fingerprint_size}"
    experiment_folder = pathlib.Path(
        model_folder, f"GraphVAE_v{version}_" + str(prog_args.dataset__num_examples)
    )

    script_path = pathlib.Path(__file__).parent
    repo_path = script_path.parent
    data_path = pathlib.Path(repo_path, "data", "QM9")
    pathlib.Path.mkdir(data_path, parents=True, exist_ok=True)

    # loading dataset
    max_num_nodes = prog_args.dataset__max_num_nodes
    dataset_split_percentages = (
        prog_args.dataset__train_percentage,
        prog_args.dataset__test_percentage,
        prog_args.dataset__val_percentage,
    )

    # LOAD DATASET QM9:
    (
        _,
        dataset_pad,
        train_dataset_loader,
        _,
        val_dataset_loader,
        max_num_nodes_dataset,
    ) = load_QM9_dataloader(
        data_path,
        max_num_nodes,
        prog_args.dataset__num_examples,
        prog_args.dataset__batch_size,
        dataset_split_list=dataset_split_percentages,
        apriori_max_num_nodes=9,
    )
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    num_nodes_features = dataset_pad[0]["features_nodes"].shape[1]
    num_edges_features = dataset_pad[0]["features_edges"].shape[1]

    training_effective_size = len(train_dataset_loader) * train_dataset_loader.batch_size
    validation_effective_size = len(val_dataset_loader) * val_dataset_loader.batch_size

    print("-------- EXPERIMENT INFO --------")

    print("Training set: {}".format(training_effective_size))
    print("Validation set : {}".format(validation_effective_size))
    print("max num nodes in dataset:", max_num_nodes_dataset)
    print("max num nodes setted:", max_num_nodes)
    print("max theoretical edges:", max_num_edges)
    print("num edges features", num_edges_features)
    print("num nodes features", num_nodes_features)

    # prima di iniziare il training creo la cartella in cui salvere i checkpoints, modello, iperparametri, etc...
    model__hyper_params = []
    dataset__hyper_params = []

    model__hyper_params.append(
        {
            "num_nodes_features": num_nodes_features,
            "num_edges_features": num_edges_features,
            "max_num_nodes": max_num_nodes,
            "max_num_edges": max_num_edges,
            "latent_dimension": prog_args.model__latent_dimension,
            "num_layers_encoder": prog_args.model__num_layers_encoder,
            "num_layers_decoder": prog_args.model__num_layers_decoder,
            "growth_factor_encoder": prog_args.model__growth_factor_encoder,
            "growth_factor_decoder": prog_args.model__growth_factor_decoder,
        }
    )

    dataset__hyper_params.append(
        {
            "learning_rate": prog_args.training__lr,
            "epochs": prog_args.training__epochs,
            "num_examples": prog_args.dataset__num_examples,
            "batch_size": prog_args.dataset__batch_size,
            "training_percentage": prog_args.dataset__train_percentage,
            "test_percentage": prog_args.dataset__test_percentage,
            "val_percentage": prog_args.dataset__val_percentage,
            "number_val_examples": validation_effective_size,
            "number_train_examples": training_effective_size,
            "loss_type": prog_args.loss__type,
            "fingerprint_model_name": prog_args.loss__fingerprint_model_name,
            "recon_w": prog_args.loss__recon_w,
            "kl_w": prog_args.loss__kl_w,
            "edge_w": prog_args.loss__edge_w,
            "node_w": prog_args.loss__node_w,
            "adj_w": prog_args.loss__adj_w,
        }
    )

    summary = Summary(experiment_folder, experiment_model_type)
    summary.save_json(model__hyper_params, file_name="hyperparams.json")
    summary.save_json(dataset__hyper_params, file_name="dataset_hyperparams.json")
    summary.save_summary_training(dataset__hyper_params, model__hyper_params, random.choice(dataset_pad))
    summary.changelog(prog_args.version, prog_args.changelog)

    neuralfingerprint_path = pathlib.Path(
        repo_path, "models_fingerprint", prog_args.loss__fingerprint_model_name
    )
    model_fingerprint = NeuralFingerprint.load_model_from_file(neuralfingerprint_path)
    NeuralFingerprint.freeze_model(model_fingerprint)
    model_fingerprint.to(device)
    model_fingerprint.eval()

    # set up the model:
    model = GraphVAE(
        latent_dim=prog_args.model__latent_dimension,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=num_nodes_features,
        num_edges_features=num_edges_features,
        num_layers_encoder=prog_args.model__num_layers_encoder,
        num_layers_decoder=prog_args.model__num_layers_decoder,
        growth_factor_encoder=prog_args.model__growth_factor_encoder,
        growth_factor_decoder=prog_args.model__growth_factor_decoder,
    )
    model = model.to(device)
    print("Model parameters: ", count_parameters(model)[0])

    loss = GraphVAE_loss(
        approx_loss,
        recon_w=prog_args.loss__recon_w,
        kl_w=prog_args.loss__kl_w,
        adj_w=prog_args.loss__adj_w,
        node_w=prog_args.loss__node_w,
        edge_w=prog_args.loss__edge_w,
    )

    print("-------- TRAINING --------")

    # training:
    train(
        model,
        train_dataset_loader,
        val_dataset_loader,
        summary,
        model_fingerprint,
        loss,
        prog_args.training__epochs,
        prog_args.training__lr,
    )

    # salvo il modello finale:
    final_model_name = "trained_GraphVAE_FINAL.pth"
    final_model_path = pathlib.Path(summary.directory_base, final_model_name)
    torch.save(model.state_dict(), final_model_path)

    # ---- INFERENCE ----
    model.eval()
    print("INFERENCE!")
    with torch.no_grad():
        generate_and_save_graphs(
            model,
            prog_args.dataset__num_examples,
            output_path=summary.directory_generated_graphs,
            treshold_adj=0.5,
            treshold_diag=0.3,
        )


if __name__ == "__main__":
    main()

import os
import csv
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import inspect
import io
import torch
import json

from datetime import datetime
from collections import defaultdict
from rdkit import rdBase

blocker = rdBase.BlockLogs()

# ---
_checkpoint_base_name = "checkpoint_"


def setup_dir(start_path: str, model_name: str = None, ohter_info: str = ""):

    data_path = os.path.join(start_path, "..", "data")
    saved_model = os.path.join(start_path, "..", "saved_model")
    if model_name:
        experiment_folder = os.path.join(saved_model, model_name + ohter_info)
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
    else:
        experiment_folder = ""

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(saved_model):
        os.makedirs(saved_model)

    return data_path, saved_model, experiment_folder


def logs_and_print(
    folder_path: str,
    log_message: str = "",
    recrate: bool = False,
    dot_line: bool = False,
    print_message: bool = True,
):
    # Verifica se la cartella esiste, altrimenti la crea
    """
    Salva un messaggio di log in un file di testo.

    Args:
        folder_path (str): Percorso della cartella in cui salvare il file di log.
        log_message (str, optional): Messaggio di log. Defaults to "".
        recrate (bool, optional): Se True, il file di log sarà  ricreato. Defaults to False.
        dot_line (bool, optional): Se True, il messaggio di log sarà una riga di punti. Defaults to False.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Nome del file log (puoi personalizzarlo)
    log_file_path = os.path.join(folder_path, "log.txt")

    # Ottieni il timestamp corrente
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepara il messaggio di log con il timestamp
    full_log_message = f"[{timestamp}] {log_message}\n"

    if recrate:
        open(log_file_path, "w").close()

    # Scrivi il messaggio nel file di log
    with open(log_file_path, "a") as log_file:
        if dot_line:
            full_log_message = "-" * 20 + "\n"
        elif print_message:
            print(log_message)

        log_file.write(full_log_message)


def generate_unique_id(params: list = [], lenght: int = 10) -> str:
    input_str = ""

    # Concateniamo le stringhe dei dati di input
    for param in params:
        if type(param) is list:
            param_1 = [str(p) if not callable(p) else p.__name__ for p in param]
        else:
            param_1 = str(param)
        input_str += str(param_1)

    # Calcoliamo il valore hash SHA-256 della stringa dei dati di input
    hash_obj = hashlib.sha256(input_str.encode())
    hex_dig = hash_obj.hexdigest()

    # Restituiamo i primi 8 caratteri del valore hash come ID univoco
    return hex_dig[:lenght]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def check_base_dir(*args):
    """
    Given a variable number of arguments, this function constructs a full path by joining the absolute path of the current file with the provided arguments.

    Args:
        *args: Variable number of arguments representing the path segments. Each argument can be a string or a list of strings. If an argument is a list, it is joined with the previous path segment using os.path.join.

    Returns:
        str: The full path constructed by joining the absolute path of the current file with the provided arguments.

    Raises:
        None

    Examples:
        >>> check_base_dir("folder1", "folder2")
        '/path/to/current/file/folder1/folder2'

        >>> check_base_dir(["folder1", "folder2"], "folder3")
        '/path/to/current/file/folder1/folder2/folder3'
    """

    # take the full path of the folder
    # absolute_path = os.path.dirname(__file__)

    caller_frame = inspect.currentframe().f_back
    caller_filename = inspect.getframeinfo(caller_frame).filename
    current_file_path = os.path.abspath(caller_filename)
    absolute_path = os.path.dirname(current_file_path)
    # print("AHHHHHH", absolute_path)

    full_path = absolute_path
    # args = [item for sublist in args for item in sublist]
    for idx, path in enumerate(args):
        # check if arguments are a list

        if type(path) is list:
            # path = [item for sublist in path for item in sublist if not isinstance(item, str)]
            for micro_path in path:
                if isinstance(micro_path, list):
                    for micro_micro_path in micro_path:
                        full_path = os.path.join(full_path, micro_micro_path)
                else:
                    full_path = os.path.join(full_path, micro_path)

        else:
            full_path = os.path.join(full_path, path)
        # print("------_::", full_path)
        # check the path exists
        if not os.path.exists(full_path):
            # print(f"Path {full_path} does not exist. Creating it...")
            os.makedirs(full_path)

    return full_path


# ------------------------------------LOGGING------------------------------------
def array_to_string(arr):
    with io.StringIO() as buffer:
        np.savetxt(buffer, arr, delimiter=",", fmt="%g")
        return buffer.getvalue().replace("\n", "|")


def log_dict_to_json(log_dict: dict, filename: str = "log.json") -> None:
    with open(filename, "w") as f:
        json.dump(log_dict, f)


def log_csv(filename: str = "log.csv", *dictionaries) -> None:
    if not dictionaries:
        raise ValueError("La lista di dizionari è vuota.")

    all_keys = set()
    for d in dictionaries:
        all_keys.update(d.keys())

    ordered_keys = ["datetime"] if "datetime" in all_keys else []
    ordered_keys.extend(sorted(k for k in all_keys if k != "datetime"))

    all_values = defaultdict(list)
    max_length = 0

    for d in dictionaries:
        for key in ordered_keys:
            if key in d:
                values = d[key]
                if isinstance(values, np.ndarray):
                    if values.ndim > 1:
                        values = [array_to_string(v) for v in values]
                    else:
                        values = [array_to_string(values)]
                elif isinstance(values, list) and values and isinstance(values[0], np.ndarray):
                    values = [array_to_string(v) for v in values]
                all_values[key].extend(values)
                max_length = max(max_length, len(all_values[key]))

    for key, value_list in all_values.items():
        if len(value_list) < max_length:
            all_values[key].extend([""] * (max_length - len(value_list)))

    file_path = pathlib.Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        writer.writerow(ordered_keys)

        for i in range(max_length):
            row = [all_values[key][i] for key in ordered_keys]
            writer.writerow(row)


def log(
    out_dir,
    model,
    discriminator,
    optimizer_G,
    params,
    g_step,
    checkpoints_dir="",
    d_loss=0,
    run_epoch=0,
    run_step=0,
    checkpoint_base_name=_checkpoint_base_name,
    accuracy=0,
):

    if run_step == params.steps:
        # quando finisco il training
        save_checkpoint(
            checkpoints_dir,
            f"model_final_{run_step}_{run_epoch}.pt",
            model,
            discriminator,
            step=run_step,
            epoch=run_epoch,
            g_step=g_step,
        )

    if run_step % params.steps_per_checkpoint == 0 and run_step > 0:
        # ogni tot mi salvo il checkpoint anche degli optimizer

        save_checkpoint(
            checkpoints_dir,
            f"{checkpoint_base_name}_s{run_step}_e{run_epoch}.pt",
            model,
            step=run_step,
            epoch=run_epoch,
            optimizer=optimizer_G,
        )

    if run_step % params.steps_per_val == 0 and run_step > 0:
        # save validation
        save_validation(
            accuracy,
            run_step,
            run_epoch,
            d_loss,
            filename=os.path.join(out_dir, "validation.csv"),
        )

    if run_step % params.steps_per_clean == 0 and run_step > 0:
        # cancello un po di checkpoint se sono troppo vecchi
        clean_old_checkpoint(checkpoints_dir, 0.5)


def save_training_metrics(data, datetime, log_file="metrics.csv"):
    if not os.path.exists(log_file):
        print("Log file not found. Log will be created by default.")
        global_path = check_base_dir("logs")
        today = datetime.now().strftime("%m-%d_%H-%M")
        log_file = os.path.join(global_path, f"metrics__{today}.csv")
        print(f"Log file path: {log_file}")

    with open(log_file, "w", newline="") as log_file:
        log_file.write("Epoch\tTrain Loss\tTrain Accuracy\tValidation Accuracy\n")
        for idx, elem in enumerate(total_elemets):
            log_file.write(
                f"{date[idx]}\t,{idx}\t,{train_total_loss[idx]:.4f},\t{train_accuracy[idx]:.4f},\t{val_accuracy[idx%len(val_accuracy)]:.4f}\n"
            )


def log_metrics(
    epochs: int,
    train_total_loss: list,
    train_dict_losses: dict = None,
    train_accuracy: list = [],
    val_accuracy: list = [],
    date: list = [],
    total_batch: int = None,
    log_file_path: str = None,
    plot_file_path: str = None,
    metric_name: str = "Validity",
    title: str = "Training and Validation Metrics",
    plot_show: bool = False,
    plot_save: bool = False,
):

    # plt.figure(figsize=(10, 6))
    plt.figure()
    freq_ticks = 5  # num of epoch to draw ticks

    if total_batch:

        num_batch_totali = epochs * total_batch

        # Crea una lista di numeri di batch
        total_elemets = list(range(1, num_batch_totali + 1))
        # Aggiungi le barre verticali per le epoche
        list_batch_epoch = []
        for epoca in range(1, epochs + 2):
            batch_inizio_epoca = (epoca - 1) * total_batch

            # if epoca % freq_ticks == 0 or epoca == 1:
            plt.axvline(x=batch_inizio_epoca, color="red", linestyle="--", alpha=0.3)
            if (epoca - 1) % freq_ticks == 0:
                list_batch_epoch.append(batch_inizio_epoca)

        plt.xticks(list_batch_epoch, [f"{epoca}" for epoca in range(0, epochs + 1, freq_ticks)])

    else:
        total_elemets = np.arange(1, epochs + 1)

    plt.plot(total_elemets, train_total_loss, label="Train Loss")

    if train_dict_losses is not None:
        plt.plot(total_elemets, train_dict_losses["train_adj_recon_loss"], label="ADJ recon Loss")
        plt.plot(total_elemets, train_dict_losses["train_kl_loss"], label="KL Loss")
        plt.plot(total_elemets, train_dict_losses["train_edge_loss"], label="Edge Loss")
        plt.plot(total_elemets, train_dict_losses["train_node_loss"], label="Node Loss")

    metric_label = "Loss"
    if val_accuracy or train_accuracy:
        metric_label += f", {metric_name}"

    if not date:
        date = [datetime.now() for _ in total_elemets]

    if not train_accuracy:
        train_accuracy = [0] * len(total_elemets)
    else:
        plt.plot(
            np.arange(1, len(total_elemets) + 1),
            train_accuracy,
            label=f"Train {metric_name}",
            marker="s",
        )
    if not val_accuracy:
        val_accuracy = [0] * len(total_elemets)
    else:
        x_val: list = [
            (epochs / len(val_accuracy) * (idx + 1)) * total_batch for idx in range(len(val_accuracy))
        ]

        plt.plot(
            x_val,
            val_accuracy,
            label=f"Validation {metric_name}",
            marker="^",
        )

    if log_file_path is None:
        print("Log file not found. Log will be created by default.")
        global_path = check_base_dir("..", "logs")
        today = datetime.now().strftime("%m-%d_%H-%M")
        log_file_path = os.path.join(global_path, f"metrics__{today}.csv")
        print(f"Log file path: {log_file_path}")

    with open(log_file_path, "w", newline="") as log_file:
        log_file.write("Epoch\tTrain Loss\tTrain Accuracy\tValidation Accuracy\n")
        for idx, elem in enumerate(total_elemets):
            log_file.write(
                f"{date[idx]}\t,{idx}\t,{train_total_loss[idx]:.4f},\t{train_accuracy[idx]:.4f},\t{val_accuracy[idx%len(val_accuracy)]:.4f}\n"
            )
    # Create the plot
    if plot_file_path is None and plot_save:
        print("Plot file not found. Plot will not be created.")
        global_path = check_base_dir("..", "logs")
        today = datetime.now().strftime("%m-%d_%H-%M")
        plot_file_path = os.path.join(global_path, f"plot__{today}.png")

    plt.xlabel("Epochs")
    plt.ylabel(metric_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if plot_save:
        plt.savefig(plot_file_path)
    if plot_show:
        plt.show()


# ------


def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd = logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp() ** 2, dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div


def to_one_hot(x, options):
    """
    Converts a tensor of values to a one-hot vector
    based on the entries in options.
    """
    return torch.nn.functional.one_hot(x.long(), len(options))


def squared_difference(input, target):
    return (input - target) ** 2


def count_parameters(model):
    """
    Counts the number of learnable and not learnable parameters in a model

    Parameters
    ----------
    model : torch.nn.Module
        The model to count the parameters of

    Returns
    -------
    learnable_params : int
        The number of learnable parameters in the model
    not_learnable_params : int
        The number of not learnable parameters in the model
    """
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    not_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    return learnable_params, not_learnable_params


def find_directory(input_path, target_dir):
    """
    Find the directory `target_dir` in the ancestors of `input_path`.

    Parameters
    ----------
    input_path : str or pathlib.Path
        The path to start searching from
    target_dir : str
        The name of the directory to search for

    Returns
    -------
    pathlib.Path
        The path of the directory `target_dir` if found, otherwise raises a ValueError

    Raises
    ------
    ValueError
        If `target_dir` is not found in the ancestors of `input_path`
    """
    path = pathlib.Path(input_path)
    while True:
        if path.name == target_dir:
            return path
        path = path.parent
        if path == pathlib.Path("/"):  # se siamo arrivati alla radice del filesystem
            raise ValueError(f"Directory '{target_dir}' non trovata")


if __name__ == "__main__":

    # Example usage:
    epochs = 10
    train_loss = [0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08, 0.07] * epochs
    train_accuracy = [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95]
    val_accuracy = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91]
    # e che conosci il numero totale di epoche e il numero di elementi per batch
    elementi_per_batch = 10

    log_metrics(epochs, train_loss, plot_show=True, elements_per_batch=elementi_per_batch)

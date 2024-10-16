import torch
import os

import re

# ------------------------------------SAVING & LOADING-----------------------------------------
# def latest_checkpoint(root_dir, base_name):
#     """
#     Find the latest checkpoint in a directory.
#     Parameters:
#         - root_dir (str): root directory
#         - base_name (str): base name of checkpoint
#     """
#     check_base_dir("..", root_dir)

#     checkpoints = [chkpt for chkpt in os.listdir(root_dir) if base_name in chkpt]
#     if len(checkpoints) == 0:
#         return None
#     latest_chkpt = ""
#     latest_step = -1
#     latest_epoch = -1
#     for chkpt in checkpoints:
#         step = torch.load(os.path.join(root_dir, chkpt))["step"]
#         epoch = torch.load(os.path.join(root_dir, chkpt))["epoch"]
#         if step > latest_step or (epoch > latest_epoch):
#             latest_epoch = epoch
#             latest_chkpt = chkpt
#             latest_step = step
#     return os.path.join(root_dir, latest_chkpt)


def clean_old_checkpoint(folder_path, percentage):
    """
    Delete old checkpoints from a directory with a given percentage.
    Parameters:
        - folder_path (str): path to directory where checkpoints are stored
        - percentage (float): percentage of checkpoints to delete

    """

    # ordina i file per data di creazione
    files = [(f, os.path.getctime(os.path.join(folder_path, f))) for f in os.listdir(folder_path)]
    files.sort(key=lambda x: x[1])

    # calcola il numero di file da eliminare
    files_to_delete = int(len(files) * percentage)

    # cicla attraverso i file nella cartella
    for file_name, creation_time in files[:-files_to_delete]:
        # costruisce il percorso completo del file
        file_path = os.path.join(folder_path, file_name)
        # elimina il file
        if file_name.endswith(".pt"):
            os.remove(file_path)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    scheduler=None,
    loss: list = None,
    other=None,
    directory: str = None,
    max_checkpoints=5,
):
    # Crea la directory se non esiste
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Definisci il nome del file del checkpoint
    checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(directory, checkpoint_name)

    # Salva il checkpoint
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint.update({"scheduler": scheduler.state_dict()})

    if loss is not None:
        checkpoint.update({"loss": loss})

    if other is not None:
        checkpoint.update({"other": other})

    torch.save(checkpoint, checkpoint_path)

    # Gestisci i vecchi checkpoint
    checkpoints = sorted(os.listdir(directory))
    # ottieni la data di creazione di ogni file
    files_with_date = [(file, os.path.getctime(os.path.join(directory, file))) for file in checkpoints]

    if len(checkpoints) > max_checkpoints:
        # Elimina il checkpoint più vecchio
        oldest_checkpoint = min(files_with_date, key=lambda x: x[1])
        os.remove(os.path.join(directory, oldest_checkpoint[0]))

    # print(f"Checkpoint salvato: {checkpoint_name}")
    return checkpoint_path


def load_latest_checkpoint(directory, model, optimizer=None, scheduler=None):
    # Controlla se la directory esiste ed è non vuota
    """
    Carica l'ultimo checkpoint salvato in una directory.

    Args:
        directory (str): Percorso della directory contenente i checkpoint.
        model (nn.Module): Modello da caricare con i parametri del checkpoint.
        optimizer (nn.Optimizer, optional): Ottimizzatore da caricare con i parametri del checkpoint. Defaults to None.
        scheduler (nn.Scheduler, optional): Scheduler da caricare con i parametri del checkpoint. Defaults to None.

    Returns:
    """

    if not os.path.exists(directory) or not os.listdir(directory):
        print("Nessun checkpoint trovato nella directory specificata.")
        return None, None

    data_saved = {}

    # Trova l'ultimo checkpoint basato sul nome del file
    checkpoints = sorted(os.listdir(directory), key=lambda x: int(re.findall("\d+", x)[0]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(directory, latest_checkpoint)

    # Carica il checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    epoch = checkpoint["epoch"]

    data_saved["epoch"] = epoch

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])

    try:
        data_saved["loss"] = checkpoint["loss"]
    except:
        data_saved["loss"] = None

    try:
        data_saved["other"] = checkpoint["other"]
    except:
        data_saved["other"] = None

    # print(f"Checkpoint caricato: {latest_checkpoint} (Epoch {epoch})")

    return epoch, data_saved


def load_checkpoint(model, optimizer, filename):
    """
    Carica un checkpoint del modello.
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"Checkpoint caricato: {filename}")
        return model, optimizer, epoch, loss
    else:
        print(f"Nessun checkpoint trovato in: {filename}")
        return model, optimizer, 0, None

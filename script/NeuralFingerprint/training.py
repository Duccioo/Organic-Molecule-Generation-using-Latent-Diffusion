import os
import numpy as np
from tqdm import tqdm
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NeuralFingerprint
import torch_geometric.transforms as T


# from rdkit.Chem import MACCSkeys
# from rdkit.Chem import Draw

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr


# ---
from data_preprocessing import (
    FilterSingleton,
    FilterMaxNodes,
    FilterErrorMolecula,
    OneHotEncoding,
    ToTensor,
    AddFIngerprint,
)
from utils import check_directory, logs_and_print
from utils.plotting import plot_loss


def load_data(root="path/to/data", size_fingerprint=1024):
    return QM9(
        root=root,
        pre_filter=T.ComposeFilters([FilterSingleton(), FilterMaxNodes(9), FilterErrorMolecula()]),
        pre_transform=T.Compose([OneHotEncoding()]),
        transform=T.Compose([AddFIngerprint(size_fingerprint), ToTensor()]),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    p_bar_batch = tqdm(loader, desc=f"Training Epoch", leave=False, position=0)
    for data in p_bar_batch:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        data.y = data.y.view(out.shape)

        loss = F.mse_loss(out, data.y)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        p_bar_batch.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def evaluate_fingerprints(true_fp, pred_fp):
    # Converti in numpy array se non lo sono già
    true_fp = np.array(true_fp)
    pred_fp = np.array(pred_fp)

    # MSE e MAE
    mse = mean_squared_error(true_fp, pred_fp)
    mae = mean_absolute_error(true_fp, pred_fp)

    # Similarità di Tanimoto (1 - distanza di Jaccard)

    tanimoto_sim = 1 - jaccard(true_fp.flatten(), pred_fp.flatten())

    # Coefficiente di correlazione di Pearson
    pearson_corr, _ = pearsonr(true_fp, pred_fp)
    pearson_corr = np.mean(pearson_corr)

    if np.isnan(pearson_corr):
        pearson_corr = 0

    # Distanza di Hamming (per fingerprint binari)
    hamming_dist = np.sum(true_fp != pred_fp) / len(true_fp)

    return {
        "MSE": mse,
        "MAE": mae,
        "Tanimoto Similarity": tanimoto_sim,
        "Pearson Correlation": pearson_corr,
        # "Hamming Distance": hamming_dist,
    }


def validate(model, loader, device):
    model.eval()
    all_true_fp = []
    all_pred_fp = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            pred_fp = model(data.x, data.edge_index, data.batch)

            data.y = data.y.view(pred_fp.shape)

            all_true_fp.extend(data.y.cpu().numpy())
            all_pred_fp.extend(pred_fp.cpu().numpy())

    # Calcola le metriche
    metrics = evaluate_fingerprints(all_true_fp, all_pred_fp)
    return metrics


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    patience=6,
    n_epochs=100,
    model_dir: str = None,
    fold_number: int = 0,
):
    log_dir = check_directory(os.path.join(model_dir, "logs"))
    logs_and_print(
        log_dir,
        f"Starting training with early stopping on fold {fold_number}",
        recrate=True,
        file_name=f"logs_fold_{fold_number}.txt",
    )
    logs_and_print(log_dir, dot_line=True, file_name=f"logs_fold_{fold_number}.txt")
    best_loss = float("inf")
    counter = 0
    train_loss_list = []
    val_metric_list = []
    for epoch in tqdm(range(n_epochs), leave=True, desc="Training loop", colour="red", position=1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step(val_metrics["MSE"])

        train_loss_list.append(train_loss)
        val_metric_list.append(val_metrics)

        logs_and_print(
            log_dir,
            f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f} "
            + f"Val MSE: {val_metrics['MSE']:.4f} "
            + f"Val MAE: {val_metrics['MAE']:.4f} "
            + f"Val Tanimoto: {val_metrics['Tanimoto Similarity']:.4f} "
            + f"Val Pearson: {val_metrics['Pearson Correlation']:.4f}",
            file_name=f"logs_fold_{fold_number}.txt",
        )
        logs_and_print(log_dir, dot_line=True, file_name=f"logs_fold_{fold_number}.txt")

        if round(val_metrics["MSE"], 4) < best_loss:
            best_loss = val_metrics["MSE"]
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logs_and_print(
                    log_dir, f"Early stopping at epoch {epoch+1}", file_name=f"logs_fold_{fold_number}.txt"
                )
                break

    # plot loss
    experiment_img = check_directory(os.path.join(model_dir, f"imgs"))
    plot_loss(train_loss_list, val_metric_list, experiment_img, fold_number)

    return best_loss, best_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # parametri
    learning_rate = 0.01
    weight_decay = 0.0005
    size_fingerprint = 2048
    n_fold = 5
    batch_size = 256
    epochs_per_fold = 100
    nn_hidden_channel = 128
    nn_layer = 5
    early_stopping_patience = 6
    version = "v2_rdkitgen"
    dataset_type = "qm9"

    experiment_name = "nn_fingerprint" + "_" + dataset_type + "_fs" + str(size_fingerprint) + "_" + version
    script_path = os.path.dirname(os.path.realpath(__file__))
    repo_path = os.path.join(script_path, os.path.pardir, os.path.pardir)
    path_data_qm9 = os.path.join(script_path, "data", "QM9")

    model_path = check_directory(os.path.join(repo_path, "models_fingerprint"))
    experiment_path = check_directory(os.path.join(model_path, experiment_name))

    # Carica e prepara i dati
    dataset = load_data(path_data_qm9, size_fingerprint=size_fingerprint)
    # data_list = prepare_data(dataset, size_fingerprint=size_fingerprint)

    # DataLoader senza shuffle
    # dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    # Converti in una lista
    data_list = []
    for data in tqdm(dataset):
        data_list.append(data)

    # salvo i parametri come json
    with open(os.path.join(experiment_path, "parameters.json"), "w") as f:
        json.dump(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "size_fingerprint": size_fingerprint,
                "n_fold": n_fold,
                "batch_size": batch_size,
                "epochs_per_fold": epochs_per_fold,
                "in_channels": dataset.num_features,
                "out_channels": size_fingerprint,
                "hidden_channels": nn_hidden_channel,
                "num_layers": nn_layer,
                "early_stopping_patience": early_stopping_patience,
                "version": version,
                "dataset_type": dataset_type,
            },
            f,
        )

    # Imposta K-Fold cross-validation
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_results = []
    best_overall_loss = float("inf")
    best_overall_model = None
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):

        if fold == 2:
            break

        print(f"Fold {fold+1}/{n_fold}")

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        # Inizializza il modello
        model = NeuralFingerprint(
            dataset.num_features,
            hidden_channels=nn_hidden_channel,
            num_layers=nn_layer,
            out_channels=size_fingerprint,
        ).to(device)

        # Ottimizzatore e scheduler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, threshold=0.005)

        # Addestramento con early stopping
        best_loss, fold_model = train_with_early_stopping(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            n_epochs=epochs_per_fold,
            patience=early_stopping_patience,
            model_dir=experiment_path,
            fold_number=fold + 1,
        )

        fold_results.append(best_loss)

        # salvo il modello per ogni fold
        fold_model_filename = os.path.join(experiment_path, f"fingerprint_model_fold{fold + 1}.pth")
        torch.save(model, fold_model_filename)

        print(f"\n\nBest validation loss for fold {fold+1}: {best_loss:.4f}")

        if best_loss < best_overall_loss:
            best_overall_loss = best_loss
            best_overall_model = fold_model
            best_fold = fold + 1

    print("---------")

    print(f"Average validation loss across folds: {sum(fold_results)/len(fold_results):.4f}")
    print(f"Best overall fold: {best_fold} with loss: {best_overall_loss:.4f}")

    # Salva il modello migliore
    model_path = os.path.join(experiment_path, f"best_fingerprint_model_fold_FINAL.pth")
    torch.save(best_overall_model, model_path)
    print(f"Best model saved as {model_path}")


if __name__ == "__main__":
    main()

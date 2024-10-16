import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import csv
from pathlib import Path
from tqdm import tqdm


def load_csv_to_arrays(filename, data_column="mean", shape=64):
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Il file {filename} non esiste.")

    data_arrays = []
    with file_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        progress_bar = tqdm(reader, desc="Loading data")
        for row in progress_bar:
            if data_column in row:
                data_row = []
                # Converte la stringa in un array NumPy
                str_data = row[data_column].strip("[]")  # Rimuove le parentesi quadre
                str_arrays = str_data.split("|")  # Divide in array separati
                del str_arrays[-1]
                len_arrays = len(str_arrays)

                len_inner = len(str_arrays[0].split(","))
                if len_inner > 1:
                    len_inner -= 1

                for str_arr in str_arrays:
                    # Converte ogni stringa in un array NumPy
                    arr = np.fromstring(str_arr, sep=",")
                    # Rimodella l'array se necessario (assumendo array 2D)
                    arr = arr.reshape(len_inner, -1).squeeze()  # Assumiamo 3 colonne, modifica se necessario
                    data_row.append(arr)

                data_arrays_row = np.array(data_row).squeeze()
                data_arrays.append(data_arrays_row)

    return data_arrays


def pca_2component(data_arrays):
    scaler = StandardScaler()
    # data_arrays = np.array(data_arrays).reshape(-1, 1)
    scaled_data = scaler.fit_transform(data_arrays)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Separa i risultati PCA per componenti
    pca_1 = pca_result[:, 0]
    pca_2 = pca_result[:, 1]

    return pca_1, pca_2


def plot_pca_distributions(
    directory: Path = None, mean_data=None, var_data=None, res_data=None, filename="pca.png"
):
    """
    Funzione per creare 1, 2 o 3 plot a seconda dei dati forniti.

    Parametri:
    mean_data (tuple): Coppia di liste o array per il primo plot (medie). Es: (mean_pca_means, mean_pca_variances).
    var_data (tuple): Coppia di liste o array per il secondo plot (varianze). Es: (var_pca_means, var_pca_variances).
    res_data (tuple): Coppia di liste o array per il terzo plot (residui). Es: (res_pca_mean, res_pca_variances).
    """

    # Controlla quanti plot fare in base ai dati forniti
    plots_to_draw = sum([mean_data is not None, var_data is not None, res_data is not None])

    # Se non ci sono dati forniti, esci dalla funzione
    if plots_to_draw == 0:
        print("Nessun dato fornito per il plotting.")
        return

    # Crea una mappa di colori personalizzata
    colors = ["blue", "green", "yellow", "red"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Configura la figura con il numero di subplot richiesti
    plt.figure(figsize=(6 * plots_to_draw, 5))

    # Plot per le medie (se fornito)
    if mean_data is not None:
        plt.subplot(1, plots_to_draw, 1)
        scatter = plt.scatter(mean_data[0], mean_data[1], c=range(len(mean_data[0])), cmap=cmap, alpha=0.7)
        plt.colorbar(scatter, label="batch_number")
        plt.title("Distribuzione PCR: Medie")
        plt.xlabel("Prima Componente Principale")
        plt.ylabel("Seconda Componente Principale")

    # Plot per le varianze (se fornito)
    if var_data is not None:
        plt.subplot(1, plots_to_draw, 2 if mean_data is not None else 1)
        scatter = plt.scatter(var_data[0], var_data[1], c=range(len(var_data[0])), cmap=cmap, alpha=0.7)
        plt.colorbar(scatter, label="batch_number")
        plt.title("Distribuzione PCR: Varianze")
        plt.xlabel("Prima Componente Principale")
        plt.ylabel("Seconda Componente Principale")

    # Plot per i residui (se fornito)
    if res_data is not None:
        plt.subplot(
            1,
            plots_to_draw,
            (
                3
                if mean_data is not None and var_data is not None
                else (2 if mean_data is not None or var_data is not None else 1)
            ),
        )
        scatter = plt.scatter(res_data[0], res_data[1], c=range(len(res_data[0])), cmap=cmap, alpha=0.7)
        plt.colorbar(scatter, label="batch_size")
        plt.title("Distribuzione PCR: Residui")
        plt.xlabel("Prima Componente Principale")
        plt.ylabel("Seconda Componente Principale")

    plt.tight_layout()

    plt.savefig(directory / filename, dpi=300)
    plt.close()


def main():

    repo_path = Path(__file__).resolve().parents[1]
    models_path = repo_path / "models"

    experiment_name = "GraphVAE_v9.3_fingerprint_fs128_50000_4"

    experiment_path = models_path / experiment_name
    log_dir = experiment_path / "logs"
    img_dir = experiment_path / "img"
    filename = "metric_training.csv"

    mean_arrays = load_csv_to_arrays(log_dir / filename, "mean")
    var_arrays = load_csv_to_arrays(log_dir / filename, "variance")

    mean_1, mean_2 = pca_2component(mean_arrays)
    var_1, var_2 = pca_2component(var_arrays)

    plot_pca_distributions(directory=img_dir, mean_data=(mean_1, mean_2), var_data=(var_1, var_2))


if __name__ == "__main__":
    main()

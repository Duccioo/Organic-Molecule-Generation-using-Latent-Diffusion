import os
import csv
import re
import pathlib

import os
import csv
import re

VERSION_DESCRIPTIONS = {
    "20.0": "Base Model",
    "20.1": "NO Fingerprint Loss",
    "20.2": "Size Fingerprint 2048",
    "20.3": "Fingerprint 128 RDkit",
    "20.4": "Approx. Loss",
    "20.5": "Latent Space = 20",
    "20.6": "Latent Space = 120",
    "20.00": "Base Model",
    "20.01": "NO Fingerprint Loss",
    "20.02": "Size Fingerprint 2048",
    "20.03": "Fingerprint 128 RDkit",
    "20.04": "Approx. Loss",
    "20.05": "Latent Space = 20",
    "20.06": "Latent Space = 120",
}


def extract_version_and_dataset_size(folder_name):
    if folder_name.startswith("Diffusion"):
        match = re.search(r"v(\d+\.\d+)_(\w+)_(\d+)_from_(\d+)", folder_name)
        if match:
            return match.group(1), match.group(2), int(match.group(3)), int(match.group(4)), "Diffusion"
    elif folder_name.startswith("GraphVAE"):
        match = re.search(r"v(\d+\.\d+)_(\w+)_(\d+)", folder_name)
        if match:
            return match.group(1), match.group(2), int(match.group(3)), None, "GraphVAE"
    return None, None, None, None, None


def extract_metrics(file_path):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                metrics[key.strip()] = value.strip()
    return metrics


def process_experiments(root_folder):
    all_data = []
    metrics_filename = "metrics_10000"
    summary_filename = "summary_metrics.txt"

    for root, dirs, files in os.walk(root_folder):
        if metrics_filename in dirs:
            summary_file = os.path.join(root, metrics_filename, summary_filename)
            if os.path.exists(summary_file):
                metrics = extract_metrics(summary_file)
                folder_name = os.path.basename(root)
                version, loss_type, dataset_size, graphvae_dataset_size, model_type = (
                    extract_version_and_dataset_size(folder_name)
                )

                if version and dataset_size:
                    row = {
                        "Model Type": model_type,
                        "Version": version,
                        "Version Description": VERSION_DESCRIPTIONS.get(version, "Unknown"),
                        "Loss Type": loss_type,
                        "Dataset Size": dataset_size,
                        "GraphVAE Dataset Size": graphvae_dataset_size,
                        **metrics,
                    }
                    all_data.append(row)

    return all_data


def write_csv(data, filename):
    if not data:
        return

    fieldnames = list(data[0].keys())
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def write_version_descriptions_csv(filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Version", "Description"])
        for version, description in VERSION_DESCRIPTIONS.items():
            writer.writerow([version, description])


def main():

    # root_folder = '\\192.168.1.104\Server_DRIFTER\UNIVERSITY\MAGISTRALE\TESI\final_models'
    root_folder_script = pathlib.Path(__file__).parent.resolve()
    root_folder = pathlib.Path("//192.168.1.104/Server_DRIFTER/UNIVERSITY/MAGISTRALE/TESI/final_models")
    output_folder = root_folder_script.parent / "result"

    os.makedirs(output_folder, exist_ok=True)

    all_data = process_experiments(root_folder)

    # Scrivi tutti i dati in un unico file CSV
    write_csv(all_data, output_folder / "all_experiments_results.csv")

    # Separa i dati per GraphVAE e Diffusion
    graphvae_data = [row for row in all_data if row["Model Type"] == "GraphVAE"]
    diffusion_data = [row for row in all_data if row["Model Type"] == "Diffusion"]

    write_csv(graphvae_data, output_folder / "graphvae_results.csv")
    write_csv(diffusion_data, output_folder / "diffusion_results.csv")

    # Scrivi la tabella delle descrizioni delle versioni
    write_version_descriptions_csv(output_folder / "version_descriptions.csv")

    print("CSV files have been created successfully.")


if __name__ == "__main__":
    main()

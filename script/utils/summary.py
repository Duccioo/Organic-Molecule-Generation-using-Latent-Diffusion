import os
import glob
import json
import pathlib
from datetime import datetime
import re

from rich import box
from rich.console import Console
from rich.table import Table
from rdkit.Chem import Draw
from rdkit import Chem


def console_bullet_list(console, list_elem: list = []):
    if isinstance(list_elem, list):
        for element in list_elem:
            console.print("- ", element)


def console_table_dict(console, dict: dict = {}, header: tuple = ("KEY", "VALUE"), inplace: bool = True):
    table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
    table.add_column(header[0])
    table.add_column(header[1])
    for key, value in dict.items():
        table.add_row(key, str(value))

    if inplace:
        console.print(table)
    return table


def console_matrix(console, matrix: list = [], inplace: bool = True):
    table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
    num_col = len(matrix[0])
    for i in range(num_col):
        table.add_column(str(i + 1))
    for row in matrix:
        row = [str(i) for i in row]
        table.add_row(*row)
    if inplace:
        console.print(table)
    return table


def check_and_create_folder(base_folder="model_saved_prova", end_of_file="FINAL.pth", overwrite=False):
    index = 0

    while True:
        current_folder = f"{base_folder}{'_' + str(index) if index > 0 else ''}"

        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
            return current_folder

        final_files = glob.glob(os.path.join(current_folder, f"*{end_of_file}"))

        if not final_files:
            print(f"Cartella {current_folder} non contiene il modello {end_of_file}")
            return current_folder
        else:
            index += 1


class Summary:
    def __init__(self, directory="model_saved_prova", model_type="GraphVAE_BASE", overwrite=False):
        self.directory_base = directory
        self.model_type = model_type

        # controllo che non ci siano cartelle con lo stesso nome
        # se ci sono, aggiungo un numero successivo facendo attenzione a controllare che il modello sia stato allenato completamente
        # ovvero che finisca per "FINAL.pth"
        self.directory_base = check_and_create_folder(base_folder=self.directory_base, overwrite=overwrite)

        self.directory_checkpoint = pathlib.Path(self.directory_base) / "checkpoints"
        pathlib.Path.mkdir(self.directory_checkpoint, parents=True, exist_ok=True)
        self.directory_log = pathlib.Path(self.directory_base) / "logs"
        pathlib.Path.mkdir(self.directory_log, parents=True, exist_ok=True)
        self.directory_generated_graphs = pathlib.Path(self.directory_base) / "generated_graphs"
        pathlib.Path.mkdir(self.directory_generated_graphs, parents=True, exist_ok=True)
        self.directory_img = pathlib.Path(self.directory_base) / "img"
        pathlib.Path.mkdir(self.directory_img, parents=True, exist_ok=True)

        self.model_architecture = pathlib.Path(self.directory_base) / "model_architecture.txt"

        self.directory_changelog_models = pathlib.Path(self.directory_base).parent
        self.changelog_filename = self.directory_changelog_models / str(
            self.model_type.replace(" ", "_") + "_changelog.md"
        )

    def save_json(self, model_params: list = [], file_name="hyperparams.json"):
        # salvo gli iperparametri:
        self.hyper_param_file = pathlib.Path(self.directory_base) / file_name
        # Caricamento dei dati dal file JSON
        with open(self.hyper_param_file, "w") as file:
            json.dump(model_params, file, indent=4)

    def save_dict_csv():
        pass

    def changelog(self, version, changes: str = ""):

        if not re.match(r"^\d+(\.\d+)?[a-zA-Z]?(_[a-zA-Z]+)?$", version):
            raise ValueError(
                "Formato versione non valido. Deve essere un numero seguito opzionalmente da un punto e altri numeri, e opzionalmente da una lettera."
            )

        # Leggi il contenuto esistente del file
        try:
            with open(self.changelog_filename, "r", encoding="utf-8") as file:
                existing_content = file.read()
        except FileNotFoundError:
            existing_content = "# Changelog\n\n"

        # Prepara il nuovo changelog
        new_changelog = f"## Versione {version}\n\n{changes}\n\n---\n\n"

        # Cerca se esiste già un changelog per questa version
        pattern = re.compile(f"## Versione {re.escape(version)}.*?---\n", re.DOTALL)
        if pattern.search(existing_content):
            # Sostituisci il changelog esistente
            updated_content = pattern.sub(new_changelog, existing_content)
            action = "sostituito"
        else:
            # Aggiungi il nuovo changelog dopo l'intestazione
            updated_content = re.sub(r"(# Changelog\n\n)", f"\\1{new_changelog}", existing_content, count=1)
            action = "aggiunto"

        # Scrivi il contenuto aggiornato nel file
        with open(self.changelog_filename, "w", encoding="utf-8") as file:
            file.write(updated_content)

        print(f"Changelog per la version {version} {action} con successo.")

    def update_json(self, new_params: dict = {}):
        # self.hyper_param_file = pathlib.Path(self.directory_base) / file_name
        try:
            with open(self.hyper_param_file, "r") as file:
                dizionario = json.load(file)
        except FileNotFoundError:
            dizionario = {}

        # Aggiunge i nuovi valori al dizionario esistente
        dizionario.update(new_params)

        with open(self.hyper_param_file, "w") as file:
            json.dump(dizionario, file, indent=4)

    def save_summary_training(
        self,
        dataset_params: list[dict] = [],
        model_params: list[dict] = [],
        example_data: dict = {},
    ):
        now = datetime.now().strftime("%Y-%m-%d %H-%M")
        filename = os.path.join(self.directory_base, "summary_experiment.md")
        with open(filename, "w") as file:
            console = Console(file=file)
            console.print("# Experiment Details")

            console.print(f"> from experiment with {self.model_type}")
            console.print(f"> on {now}")
            console.print("## Model")
            try:
                console_table_dict(console, model_params[0])
            except:
                console_table_dict(console, model_params)
            console.print("## Dataset")
            console.print(
                "- Dataset used [QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html)"
            )
            console_table_dict(console, dataset_params[0])

            self.save_example_data(console, example_data)

    def save_example_data(self, console: Console, example_data, details=True):
        console.print("## Example data")
        summary = "Example Data from QM9 dataset Padded"
        console.print(f"<details><summary>{summary}</summary>\n")
        for key, value in example_data.items():

            if isinstance(value, int) or isinstance(value, str):
                console.print(f"#### {key} :\n - {str(value)}")
                if key == "smiles":
                    mol = Chem.MolFromSmiles(str(value))
                    mol_filepath = os.path.join(self.directory_base, "example_molecule.png")
                    Draw.MolToFile(mol, mol_filepath)
                    console.print("\n<img src='example_molecule.png'>")

            else:
                console.print("#### " + key + " :")
                console.print("> __SHAPE__ : " + str(value.shape))
                table = console_matrix(console, value.tolist())
        console.print("</details>")


if __name__ == "__main__":
    with open("prova.md", "w") as file:
        console = Console(file=file)

        table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
        table.add_column("Released", justify="center", style="cyan", no_wrap=True)
        table.add_column("Title", justify="center", style="magenta")

        table.add_row("Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$952,110,690")
        table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
        table.add_row("Dec 15, 2017", "Star Wars Ep. V111: The Last Jedi", "$1,332,539,889")
        for i in range(3):
            table.add_row("Dec 16, 2016", "Rogue One: A Star Wars Story", f"{i*100}")

        console.print("## INFORMATION", style="blue")
        console.print(table)
        console.print("## DANGER!", style="red on white")
        console.print_json(data={"prova": "proav1"})
        lista = ["prova1", "prova2", "prova3"]
        console_bullet_list(console, lista)

        dicta = {"prova1": "prova1", "prova2": "prova2", "prova3": "prova3"}
        console_table_dict(console, dicta)

        matrix = [["ppp", "bbbb", "zzzz"], ["ppp", "bbbb"], ["ppp", "bbbb"]]
        prova = np.ones((3, 7))
        console_matrix(console, prova)

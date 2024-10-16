import argparse
import os
from datetime import datetime
import json

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from rdkit.Chem import Draw
from tqdm import tqdm

# ---
from GraphVAE.model_graphvae import GraphVAE
from data_graphvae import load_QM9
from utils import (
    load_from_checkpoint,
    save_checkpoint,
    log_metrics,
    latest_checkpoint,
    set_seed,
    graph_to_mol,
    generate_unique_id,
)


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument(
        "--hyper_file",
        type=str,
        help="Path al file json degli iperparametri",
        default="",
    )
    parser.add_argument(
        "--model_file", type=str, help="Path al file del modello finale", default=""
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path alla cartella dove salvare encoder e decoder",
        default="encoder_decoder_saved",
    )

    return parser.parse_args()


if __name__ == "__main__":

    arg_parsed = arg_parse()

    # Path al json degli iperparametri
    json_path = arg_parsed.hyper_file

    with open(json_path, "r") as file:
        dati = json.load(file)
        dati = dati[0]

    # Path al modello finale salvato
    model_path = arg_parsed.model_file

    model = GraphVAE(
        dati["max_num_nodes"],
        dati["latent_dimension"],
        max_num_nodes=dati["max_num_nodes"],
        max_num_edges=dati["max_num_edges"],
        num_nodes_features=dati["num_nodes_features"],
        num_edges_features=dati["num_edges_features"],
    )

    data_model = torch.load(model_path, map_location="cpu")
    model.load_state_dict(data_model)

    model.save_vae_decoder(arg_parsed.output_path)
    model.save_vae_encoder(arg_parsed.output_path)

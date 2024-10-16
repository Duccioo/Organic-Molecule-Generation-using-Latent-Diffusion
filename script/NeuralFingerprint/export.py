import os
import json
from torch_geometric.nn import NeuralFingerprint
import torch


# ---


def load_model_from_file(path: str):
    """
    Loads a NeuralFingerprint model from a given directory path.

    This function first searches for a JSON file in the given directory path. If a JSON file is found, it is assumed
    to contain the hyperparameters of the model to be loaded. The hyperparameters are used to create an instance of
    NeuralFingerprint. Then, the function searches for a PTH file in the same directory path. If a PTH file is found, it
    is assumed to contain the saved state dictionary of the model. The state dictionary is loaded into the model
    instance.

    Args:
        path: The directory path where the JSON and PTH files are to be found.

    Returns:
        The loaded model instance if both a JSON and a PTH file are found, otherwise None.
    """
    json_files = [file for file in os.listdir(path) if file.endswith(".json")]

    if json_files:
        hyper_file_json = os.path.join(path, json_files[0])

        with open(hyper_file_json, "r") as file:
            hyper_params = json.load(file)

        print(hyper_file_json)

        if isinstance(hyper_params, list):
            hyper_params = hyper_params[0]

        input_channels = hyper_params["in_channels"]
        hidden_channels = hyper_params["hidden_channels"]
        out_channels = hyper_params["out_channels"]
        num_layers = hyper_params["num_layers"]

        model_fingerprint = NeuralFingerprint(
            input_channels, hidden_channels, out_channels, num_layers
        )

        # load the state dict
        model_saved = [file for file in os.listdir(path) if file.endswith("FINAL.pth")]
        if model_saved:
            model_fingerprint.load_state_dict(
                torch.load(
                    os.path.join(path, model_saved[0]),
                    map_location="cpu",
                    weights_only=True,
                )
            )

            return model_fingerprint
        else:
            print("No .pth file found, impossible to load model")
            return None

    else:
        print("No .json file found, impossible to load hyperparameters")
        return None


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

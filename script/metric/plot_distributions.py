# coding=utf-8
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as pat
import pathlib
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument(
        "--gen_path",
        type=str,
        dest="generated_graph_path",
        help="path to generated graphs",
    )
    parser.add_argument("--model_folder", type=str, help="path to model folder")
    parser.add_argument("--num_samples", type=int, help="Number of samples")

    parser.set_defaults(
        generated_graph_path="generated_graphs_v5_approx",
        model_folder="GraphVAE_v9.1_fingerprint_fs128_50000",
    )
    return parser.parse_args()


def setup_directory(model_folder_name: str) -> pathlib.Path:
    model_name = model_folder_name
    script_path = pathlib.Path(__file__)
    repo_path = script_path.parent
    saved_models_directory = repo_path / "models"
    model_directory = saved_models_directory / model_name

    metric_folder = model_directory / "metrics" / "stats"
    output_folder = model_directory / "metrics" / "plots"
    # check if output folder exists
    os.makedirs(output_folder, exist_ok=True)

    return output_folder


def plot_distributions(
    distribution_list,
    output_folder,
    distribution_names=["Dataset", "Generated"],
    bin_count=20,
    plot_name: str = "",
):
    # build value dictionaries
    Y_dict = dict()
    X_dict = dict()
    # prepare figure
    figure = plt.figure(figsize=(8, 6))
    figure.subplots(2, sharex=True, sharey=True)
    # set figure title
    # figure.suptitle("--__--", fontsize=16)
    # retrieve list of axes in the figure
    axes = figure.get_axes()
    # find minimum and maximum x values
    x_min = None
    x_max = None
    for distribution in distribution_list:
        # load the array of scores from file

        if x_min is None:
            x_min = distribution.min()
        elif x_min > distribution.min():
            x_min = distribution.min()
        if x_max is None:
            x_max = distribution.max()
        elif x_max < distribution.max():
            x_max = distribution.max()
    # plot the distributions
    for index, distribution in enumerate(distribution_list):
        # load the array of scores from file

        # plot as a histogram
        Y, X, patches = axes[index].hist(distribution, bins=bin_count, range=(x_min, x_max), density=True)
        # save X and Y for later use
        key = plot_name + "_" + str(index)
        Y_dict[key] = Y
        X_dict[key] = X
        # add axis label
        axes[index].set_title(distribution_names[index])
    # save figure
    plt.savefig(str(output_folder / "histograms_") + plot_name + ".png", dpi=100)
    plt.close()

    # prepare figure
    figure = plt.figure(figsize=(8, 6))
    # set figure title
    # figure.suptitle(plots[p], fontsize=16)
    # build arrays of style features
    colors = ["blue", "red", "yellow", "green", "grey", "black"]
    markers = ["", "", "", "", "", ""]
    # plot the distributions
    for index, distribution in enumerate(distribution_list):
        # retrieve X and Y values to plot
        key = plot_name + "_" + str(index)
        Y = Y_dict[key]
        X = X_dict[key]
        # delete redundant X value (origin)
        X = np.delete(X, 0)
        # plot as a line
        line = plt.plot(Y, color=colors[index], marker=markers[index])
    # build list of proxy artists for the legend
    proxy_artists = list()
    for index, distribution in enumerate(distribution_list):
        proxy_artists.append(pat.Patch(color=colors[index], label=distribution))
    # build legend on the axes object
    axes = figure.get_axes()[0]
    axes.legend(
        handles=(pa for pa in proxy_artists), labels=(d for d in distribution_names), loc="upper right"
    )
    # save figure
    plt.savefig(str(output_folder / "lines_") + plot_name + ".png", dpi=100)
    plt.close()


def main() -> None:
    parser = arg_parse()

    model_name = parser.model_folder
    script_path = pathlib.Path(__file__)
    repo_path = script_path.parent
    saved_models_directory = repo_path / "models"
    model_directory = saved_models_directory / model_name

    metric_folder = model_directory / "metrics" / "stats"
    output_folder = model_directory / "metrics" / "plots"
    # check if output folder exists
    os.makedirs(output_folder, exist_ok=True)
    plots = ["QED score", "logP score", "Mol. weight"]
    plot_names = ["QED_scores", "logP_scores", "molecular_weights"]
    distributions = ["Ours (C2)", "Ours (C3)", "Training", "Test"]
    dist_names = ["V4g1", "V5g4", "QM9_Training_Set", "QM9_Test_Set"]
    bin_count = 20

    # build value dictionaries
    Y_dict = dict()
    X_dict = dict()
    # plot the histograms of each score in a separate file
    for p in range(len(plots)):
        # prepare figure
        figure = plt.figure(figsize=(8, 6))
        figure.subplots(2, 2, sharex=True, sharey=True)
        # set figure title
        figure.suptitle(plots[p], fontsize=16)
        # retrieve list of axes in the figure
        axes = figure.get_axes()
        # find minimum and maximum x values
        x_min = None
        x_max = None
        for d in range(len(distributions)):
            # load the array of scores from file
            path = metric_folder / dist_names[d] / plot_names[p] + ".txt"
            array = np.loadtxt(path, delimiter=",")
            if x_min is None:
                x_min = array.min()
            elif x_min > array.min():
                x_min = array.min()
            if x_max is None:
                x_max = array.max()
            elif x_max < array.max():
                x_max = array.max()
        # plot the distributions
        for d in range(len(distributions)):
            # load the array of scores from file
            path = metric_folder / dist_names[d] / plot_names[p] + ".txt"
            array = np.loadtxt(path, delimiter=",")
            # plot as a histogram
            Y, X, patches = axes[d].hist(array, bins=bin_count, range=(x_min, x_max), density=True)
            # save X and Y for later use
            key = plots[p] + " " + distributions[d]
            Y_dict[key] = Y
            X_dict[key] = X
            # add axis label
            axes[d].set_title(distributions[d])
        # save figure
        plt.savefig(output_folder / "histograms_" + plot_names[p] + ".png", dpi=100)

    # plot the normalized distributions as overlapping curves
    for p in range(len(plots)):
        # prepare figure
        figure = plt.figure(figsize=(8, 6))
        # set figure title
        figure.suptitle(plots[p], fontsize=16)
        # build arrays of style features
        colors = ["yellow", "green", "blue", "red", "grey", "black"]
        markers = ["", "", "", "", "", ""]
        # plot the distributions
        for d in range(len(distributions)):
            # retrieve X and Y values to plot
            key = plots[p] + " " + distributions[d]
            Y = Y_dict[key]
            X = X_dict[key]
            # delete redundant X value (origin)
            X = np.delete(X, 0)
            # plot as a line
            line = plt.plot(X, Y, color=colors[d], marker=markers[d])
        # build list of proxy artists for the legend
        proxy_artists = list()
        for d in range(len(distributions)):
            proxy_artists.append(pat.Patch(color=colors[d], label=distributions[d]))
        # build legend on the axes object
        axes = figure.get_axes()[0]
        axes.legend(
            handles=(pa for pa in proxy_artists), labels=(d for d in distributions), loc="upper right"
        )
        # save figure
        plt.savefig(output_folder / "lines_" + plot_names[p] + ".png", dpi=100)


if __name__ == "__main__":
    main()

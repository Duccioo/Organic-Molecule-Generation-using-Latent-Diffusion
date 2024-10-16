import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy
import pathlib


def frechet_distance(array_A, array_B, bin_count=20, density="density"):
    # determine density truth value
    use_density = None
    if density == "density":
        use_density = True
    elif density == "count":
        use_density = False
    else:
        sys.exit("ERROR: unknown mode |" + density + "|, should be either |density| or |count|")

    # retrieve min and max values
    amax = np.max(array_A)
    bmax = np.max(array_B)
    amin = np.min(array_A)
    bmin = np.min(array_B)

    max_ = max(amax, bmax)
    min_ = min(amin, bmin)
    # bin arrays

    # get histogram and bin
    histogram, edges = np.histogram(array_A, bins=bin_count, density=use_density, range=(min_, max_))
    # calculate X values as the intermediate points between two consecutive bin edges
    X = list()
    for j in range(bin_count):
        X.append(float(edges[j] + edges[j + 1]) / 2)
    X = np.array(X, ndmin=2)
    # trasnform Y array into a probability distribution
    Y = np.divide(histogram, np.sum(histogram))
    Y = np.array(Y, ndmin=2)
    # concatenate X and Y
    V = np.transpose(np.concatenate((X, Y), axis=0))
    hist_A = V
    # process B distribution
    # get histogram and bin edges
    histogram, edges = np.histogram(array_B, bins=bin_count, density=use_density, range=(min_, max_))
    # calculate X values as the intermediate points between two consecutive bin edges
    X = list()
    for j in range(bin_count):
        X.append(float(edges[j] + edges[j + 1]) / 2)
    X = np.array(X, ndmin=2)
    # trasnform Y array into a probability distribution
    Y = np.divide(histogram, np.sum(histogram))
    Y = np.array(Y, ndmin=2)
    # concatenate X and Y
    V = np.transpose(np.concatenate((X, Y), axis=0))
    hist_B = V

    # calculate Frechet distance for each measure between A and B
    # build distance matrix between distribution A and distribution B
    distance_matrix = scipy.spatial.distance_matrix(hist_A, hist_B, p=2)
    # calculate Frechet distance as the maximum over all the points p1 in A of the minimum distance between p1 and any point p2 in B
    minimum_distances = distance_matrix.min(axis=0)
    frechet_distance = minimum_distances.max()

    return frechet_distance


def frechet_distance_old(arrays_A, arrays_B, use_density, density, measures, bin_count):

    measures_names = [m.replace(" ", "_").lower() for m in measures]

    # determine density truth value
    use_density = None
    if density == "density":
        use_density = True
    elif density == "count":
        use_density = False
    else:
        sys.exit("ERROR: unknown mode |" + density + "|, should be either |density| or |count|")

    # retrieve min and max values
    max_list = list()
    min_list = list()
    for m in range(len(measures)):
        amax = np.max(arrays_A[m])
        bmax = np.max(arrays_B[m])
        amin = np.min(arrays_A[m])
        bmin = np.min(arrays_B[m])
        if amax > bmax:
            max_list.append(amax)
        else:
            max_list.append(bmax)
        if amin < bmin:
            min_list.append(amin)
        else:
            min_list.append(bmin)

    # bin arrays
    hist_A = list()
    # process A distribution
    for i in range(len(arrays_A)):
        # get histogram and bin edges
        histogram, edges = np.histogram(
            arrays_A[i], bins=bin_count, density=use_density, range=(min_list[i], max_list[i])
        )
        # calculate X values as the intermediate points between two consecutive bin edges
        X = list()
        for j in range(bin_count):
            X.append(float(edges[j] + edges[j + 1]) / 2)
        X = np.array(X, ndmin=2)
        # trasnform Y array into a probability distribution
        Y = np.divide(histogram, np.sum(histogram))
        Y = np.array(Y, ndmin=2)
        # concatenate X and Y
        V = np.transpose(np.concatenate((X, Y), axis=0))
        # æppend histogram to list
        hist_A.append(V)
    # process B distribution
    hist_B = list()
    for i in range(len(arrays_B)):
        # get histogram and bin edges
        histogram, edges = np.histogram(
            arrays_B[i], bins=bin_count, density=use_density, range=(min_list[i], max_list[i])
        )
        # calculate X values as the intermediate points between two consecutive bin edges
        X = list()
        for j in range(bin_count):
            X.append(float(edges[j] + edges[j + 1]) / 2)
        X = np.array(X, ndmin=2)
        # trasnform Y array into a probability distribution
        Y = np.divide(histogram, np.sum(histogram))
        Y = np.array(Y, ndmin=2)
        # concatenate X and Y
        V = np.transpose(np.concatenate((X, Y), axis=0))
        # æppend histogram to list
        hist_B.append(V)

    # calculate Frechet distance for each measure between A and B
    for m in range(len(measures)):
        # build distance matrix between distribution A and distribution B
        distance_matrix = scipy.spatial.distance_matrix(hist_A[m], hist_B[m], p=2)
        # calculate Frechet distance as the maximum over all the points p1 in A of the minimum distance between p1 and any point p2 in B
        minimum_distances = distance_matrix.min(axis=0)
        frechet_distance = minimum_distances.max()
        # print Frechet distance
        print("Frechet Distance on " + measures[m] + " = " + str(frechet_distance))


def main() -> None:
    # parameters
    directory = "StatsFolders/"
    measures = ["QED score", "logP score", "Molecular Weight", "Ring Count"]
    measure_names = ["QED_scores", "logP_scores", "molecular_weights", "ring_counts"]

    dist_names = [sys.argv[1], sys.argv[2]]
    density = sys.argv[3]
    bin_count = 20
    # load stats
    arrays_A = list()
    for mn in measure_names:
        arrays_A.append(np.loadtxt(directory + dist_names[0] + "/" + mn + ".txt", delimiter=","))
    arrays_B = list()
    for mn in measure_names:
        arrays_B.append(np.loadtxt(directory + dist_names[1] + "/" + mn + ".txt", delimiter=","))

    # calculate Frechet distance
    for m in range(len(measures)):
        print(frechet_distance(arrays_A[m], arrays_B[m], bin_count, density))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# 3d41c24d-1e20-459e-ab2e-5f0e184f26aa  --  Jose Mataix Perez
# 5ca5e08e-855f-41d0-9025-06918d611fd2  --  Antonio Trujillo Reino
# e463771c-c409-4c11-b74f-687823d73cc2  --  Pau Azpeitia

# K-Means clustering for 2D data.
# Groups data into a specified number of clusters.
# Supports random and k-means++ initialization.

import argparse
import numpy as np
import sklearn.datasets
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", choices=["random", "kmeans++"], help="Initialization")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perform")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.
def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        plt.gcf().get_axes() or plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

def main(args: argparse.Namespace) -> np.ndarray:
    generator = np.random.RandomState(args.seed)

    # Generate an artificial dataset.
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    if args.init == "random":
        indices = generator.choice(len(data), size=args.clusters, replace=False)
        centers = data[indices]
    elif args.init == "kmeans++":
        centers = []
        first_index = generator.randint(len(data))
        centers.append(data[first_index])

        for _ in range(1, args.clusters):
            distances = np.min([np.linalg.norm(data - center, axis=1)**2 for center in centers], axis=0)
            probabilities = distances / np.sum(distances)
            next_index = generator.choice(len(data), p=probabilities)
            centers.append(data[next_index])

        centers = np.array(centers)

    if args.plot:
        plot(args, 0, data, centers, clusters=None)


    for iteration in range(args.iterations):
        
        distances = np.array([np.linalg.norm(data - center, axis=1) for center in centers])
        clusters = np.argmin(distances, axis=0)

       #actualiza centroides
        new_centers = np.array([data[clusters == i].mean(axis=0) for i in range(args.clusters)])

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters

if __name__ == "__main__":

    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    clusters = main(main_args)
    print("Cluster assignments:", clusters, sep="\n")

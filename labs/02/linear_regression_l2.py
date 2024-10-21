#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()
    X = dataset.data
    Y = dataset.target

    # Split the dataset into a train set and a test set.
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed
    )

    lambdas = np.geomspace(0.01, 10, num=500)
    rmses = []

    # Fit the train set using L2 regularization (Ridge) and compute RMSE.
    for lambda_ in lambdas:
        model = sklearn.linear_model.Ridge(alpha=lambda_)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(Y_test, Y_pred))
        rmses.append(rmse)

    # Find the lambda with the lowest RMSE.
    best_index = np.argmin(rmses)
    best_lambda = lambdas[best_index]
    best_rmse = rmses[best_index]

    if args.plot:
        # Plot the RMSE against lambda values.
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength $\\lambda$")
        plt.ylabel("RMSE")
        plt.title("RMSE vs. Lambda for Ridge Regression")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(main_args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))

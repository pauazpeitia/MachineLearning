#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt  # Importing for plotting

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # Append a constant feature with value 1 to the end of every input data.
    data = np.hstack((data, np.ones((data.shape[0], 1))))  # Add bias term

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed
    )

    # Generate initial linear regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # Process the data in the order of `permutation`.
        for i in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[i:i + args.batch_size]
            x_batch = train_data[batch_indices]
            y_batch = train_target[batch_indices]

            # Compute the predictions
            predictions = x_batch @ weights  # Matrix multiplication
            
            # Compute the gradients
            error = predictions - y_batch
            gradient = x_batch.T @ error / args.batch_size  # Average gradient
            
            # Apply L2 regularization (excluding bias)
            l2_gradient = np.zeros_like(weights)
            l2_gradient[:-1] = args.l2 * weights[:-1]  # Regularize only weights, not the bias

            # Update the weights
            weights -= args.learning_rate * (gradient + l2_gradient)

        # Calculate RMSE for train and test datasets
        train_rmse = np.sqrt(sklearn.metrics.mean_squared_error(train_target, train_data @ weights))
        test_rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_target, test_data @ weights))
        
        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

    # Compute RMSE for explicit Linear Regression
    explicit_model = sklearn.linear_model.LinearRegression()
    explicit_model.fit(train_data[:, :-1], train_target)  # Ignore bias column for fitting
    explicit_rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_target, explicit_model.predict(test_data[:, :-1])))

    if args.plot:
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")

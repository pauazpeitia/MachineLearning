#!/usr/bin/env python3
<<<<<<< HEAD

#5ca5e08e-855f-41d0-9025-06918d611fd2  --  Antonio Trujillo Reino
#3d41c24d-1e20-459e-ab2e-5f0e184f26aa  --  Jose Mataix Perez
#e463771c-c409-4c11-b74f-687823d73cc2  --  Pau Azpeitia

=======
>>>>>>> 7c8f32b4746c65d78016187284e728151e21c4bb
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
<<<<<<< HEAD
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
=======
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
>>>>>>> 7c8f32b4746c65d78016187284e728151e21c4bb
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

<<<<<<< HEAD
    # TODO: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.

    dataNew = np.hstack([data, np.ones((data.shape[0], 1))])

=======
    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.

>>>>>>> 7c8f32b4746c65d78016187284e728151e21c4bb
    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

<<<<<<< HEAD
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataNew, target, test_size=args.test_size, random_state=args.seed)

=======
>>>>>>> 7c8f32b4746c65d78016187284e728151e21c4bb
    # Generate initial logistic regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

<<<<<<< HEAD
        X = train_data[permutation]
        y = train_target[permutation]
        b = args.batch_size
        for i in range(int(len(train_data)/b)):
            gradient_sum = 0
            for j in range(int(b)):
                t = np.divide(1,1+np.exp(-(np.dot(X[b*i+j], weights))))
                gradient_sum += (t - y[b*i+j])*X[b*i+j]
            gradient_sum /= b
        
            weights -= args.learning_rate * gradient_sum

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        
        y_train = [np.divide(1,1+np.exp(-(np.dot(train_data[i], weights)))) for i in range(np.shape(train_data)[0])]
        y_test = [np.divide(1,1+np.exp(-(np.dot(test_data[i], weights)))) for i in range(np.shape(test_data)[0])]
        
        train_pred = []
        test_pred = []

        for i in range(np.shape(train_data)[0]):
            if y_train[i] >= 0.5: train_pred.append(1)
            else: train_pred.append(0)

        for i in range(np.shape(test_data)[0]):
            if y_test[i] >= 0.5: test_pred.append(1)
            else: test_pred.append(0)

        #Loss        
        train_loss = sklearn.metrics.log_loss(train_target, y_train)
        test_loss = sklearn.metrics.log_loss(test_target, y_test)

        #Accuracy
        train_hits = 0
        for i in range(np.shape(train_data)[0]):
            if(train_pred[i] == train_target[i]):
                train_hits += 1

        test_hits = 0
        for i in range(np.shape(test_data)[0]):
            if(test_pred[i] == test_target[i]):
                test_hits += 1

        train_accuracy = np.divide(train_hits, np.shape(train_data)[0])
        test_accuracy = np.divide(test_hits, np.shape(test_data)[0])
        
=======
        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        train_accuracy, train_loss, test_accuracy, test_loss = ...

>>>>>>> 7c8f32b4746c65d78016187284e728151e21c4bb
        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=20, cmap="RdBu", alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, label="train", marker="P", cmap="RdBu")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap="RdBu")
            plt.legend(loc="upper right")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))

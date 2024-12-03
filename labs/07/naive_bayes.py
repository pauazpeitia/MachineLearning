#!/usr/bin/env python3

#3d41c24d-1e20-459e-ab2e-5f0e184f26aa  --  Jose Mataix Perez
#5ca5e08e-855f-41d0-9025-06918d611fd2  --  Antonio Trujillo Reino
#e463771c-c409-4c11-b74f-687823d73cc2  --  Pau Azpeitia

import argparse
import numpy as np
import scipy.stats
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.

    # Compute the class prior
    class_prior = np.bincount(train_target) / len(train_target)

    if args.naive_bayes_type == "gaussian":
        # Gaussian NB: Estimate mean and variance for each class
        means = np.array([train_data[train_target == c].mean(axis=0) for c in range(args.classes)])
        variances = np.array([train_data[train_target == c].var(axis=0) for c in range(args.classes)]) + args.alpha

        def predict(data):
            log_probs = []
            for c in range(args.classes):
                # Compute log P(class)
                log_prior = np.log(class_prior[c])
                # Compute log P(x | class) using Gaussian likelihood
                log_likelihood = np.sum(
                    scipy.stats.norm.logpdf(data, loc=means[c], scale=np.sqrt(variances[c])), axis=1)
                log_probs.append(log_prior + log_likelihood)
            return np.vstack(log_probs).T

    elif args.naive_bayes_type == "multinomial":
        # Multinomial NB: Estimate probabilities with smoothing
        class_counts = np.array([train_data[train_target == c].sum(axis=0) for c in range(args.classes)])
        smoothed_counts = class_counts + args.alpha
        class_probabilities = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)

        def predict(data):
            log_probs = []
            for c in range(args.classes):
                # Compute log P(class)
                log_prior = np.log(class_prior[c])
                # Compute log P(x | class) using multinomial likelihood
                log_likelihood = data @ np.log(class_probabilities[c])
                log_probs.append(log_prior + log_likelihood)
            return np.vstack(log_probs).T

    elif args.naive_bayes_type == "bernoulli":
        # Bernoulli NB: Binarize features as [feature_value >= 8]
        binarized_train = (train_data >= 8).astype(int)

        # Estimate Bernoulli probabilities with smoothing
        class_feature_counts = np.array([binarized_train[train_target == c].sum(axis=0) for c in range(args.classes)])
        class_totals = np.array([(train_target == c).sum() for c in range(args.classes)])
        smoothed_probabilities = (class_feature_counts + args.alpha) / (class_totals[:, None] + 2 * args.alpha)

        def predict(data):
            # Binarize test data
            binarized_data = (data >= 8).astype(int)
            log_probs = []
            for c in range(args.classes):
                # Compute log P(class)
                log_prior = np.log(class_prior[c])
                # Compute log P(x | class) using Bernoulli likelihood
                log_likelihood = (
                    binarized_data * np.log(smoothed_probabilities[c]) +
                    (1 - binarized_data) * np.log(1 - smoothed_probabilities[c])
                ).sum(axis=1)
                log_probs.append(log_prior + log_likelihood)
            return np.vstack(log_probs).T

    # TODO: Predict the test data classes, and compute
    # - the test set accuracy, and
    # - the joint log-probability of the test set, i.e.,
    #     \sum_{(x_i, t_i) \in test set} \log P(x_i, t_i).
    log_probs = predict(test_data)
    predictions = np.argmax(log_probs, axis=1)

    # Test set accuracy
    test_accuracy = np.mean(predictions == test_target)

    # Joint log-probability
    test_log_probability = np.sum([log_probs[i, test_target[i]] for i in range(len(test_target))])

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(main_args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))

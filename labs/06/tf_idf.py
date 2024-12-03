#!/usr/bin/env python3

#3d41c24d-1e20-459e-ab2e-5f0e184f26aa  --  Jose Mataix Perez
#5ca5e08e-855f-41d0-9025-06918d611fd2  --  Antonio Trujillo Reino
#e463771c-c409-4c11-b74f-687823d73cc2  --  Pau Azpeitia

import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import re

from collections import Counter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=79, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

def extract_features(document, term_to_idx):
        features_vector = np.zeros(len(term_to_idx), dtype=np.float32)
        doc_term_freqs = Counter(re.findall(r'\w+', document))
        for word, freq in doc_term_freqs.items():
            if word in term_to_idx:
                features_vector[term_to_idx[word]] = freq
        return features_vector

def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.

    counter = Counter()
    for i in train_data:
        counter.update(re.findall(r'\w+', i))

    frequent_terms = {word for word, freq in counter.items() if freq >= 2}
    term_to_idx = {word: idx for idx, word in enumerate(sorted(frequent_terms))}

    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to counts but normalized to sum to 1);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    X_train_features = np.array([extract_features(doc, term_to_idx) for doc in train_data])
    X_test_features = np.array([extract_features(doc, term_to_idx) for doc in test_data])

    if args.tf:
        train_sums = np.sum(X_train_features, axis=1, keepdims=True)
        X_train_features = np.divide(X_train_features, train_sums, out=np.zeros_like(X_train_features), where=train_sums != 0)
        test_sums = np.sum(X_test_features, axis=1, keepdims=True)
        X_test_features = np.divide(X_test_features, test_sums, out=np.zeros_like(X_test_features), where=test_sums != 0)
    
    if not args.tf:
        X_train_features = (X_train_features > 0).astype(float)
        X_test_features = (X_test_features > 0).astype(float)
    
    if args.idf:
        doc_freqs = np.sum(X_train_features > 0, axis=0)
        idf_values = np.log(len(train_data) / (doc_freqs + 1))
        X_train_features *= idf_values
        X_test_features *= idf_values

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)`
    # model on the train set, and classify the test set.

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    model.fit(X_train_features, train_target)

    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    predictions = model.predict(X_test_features)
    f1_score = sklearn.metrics.f1_score(test_target, predictions, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(main_args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(main_args.tf, main_args.idf, f1_score))

#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")

class Dataset:
    def __init__(self, name="mnist.train.npz", data_size=None, url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model
        np.random.seed(args.seed)
        train = Dataset()

        # Train the model (e.g., MLPClassifier)
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, random_state=args.seed)
        model.fit(train.data, train.target)

        # Serialize the trained model with compression
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # We are predicting
        test = Dataset(args.predict)

        # Load the model
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate predictions
        predictions = model.predict(test.data)

        return predictions

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

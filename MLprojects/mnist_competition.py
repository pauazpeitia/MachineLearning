#!/usr/bin/env python3

#3d41c24d-1e20-459e-ab2e-5f0e184f26aa  --  Jose Mataix Perez
#5ca5e08e-855f-41d0-9025-06918d611fd2  --  Antonio Trujillo Reino
#e463771c-c409-4c11-b74f-687823d73cc2  --  Pau Azpeitia

# Multi-layer Perceptron (MLP) for MNIST digit classification.
# Uses GridSearchCV for hyperparameter tuning (learning rate, solver, etc.).
# Trains a model and saves it for future predictions.

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.pipeline
import sklearn.metrics
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")

class Dataset:
    """MNIST Dataset.
    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., data and optionally target.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)
        self.data /= 255.0


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        SS_transformer = sklearn.preprocessing.StandardScaler()

        # Using a Multi-layer Perceptron (MLP) classifier, a type of neural network
        mlp_model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=args.seed)
        pipeline = sklearn.pipeline.Pipeline([
            ('scaler', SS_transformer),
            ('mlp', mlp_model)
        ])

        # Hyperparameter tuning with GridSearchCV for MLP
        param_grid = {
            'mlp__alpha': [0.0001, 0.001, 0.01],  # L2 regularization strength
            'mlp__learning_rate': ['constant', 'adaptive'],
            'mlp__solver': ['adam', 'sgd'],  # Optimizer choice
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(train.data, train.target)
        # Best model found during grid search
        best_model = grid_search.best_estimator_

        # Evaluate the model on the training set
        train_predictions = best_model.predict(train.data)
        accuracy = sklearn.metrics.accuracy_score(train.target, train_predictions)
        print(f"Entrenamiento - Precisión: {accuracy * 100:.2f}%")

        # Save the trained model
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(best_model, model_file)

    else:
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

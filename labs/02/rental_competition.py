#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
import urllib.request

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--alpha", default=1.0, type=float, help="Regularization strength for Ridge")

class Dataset:
    """Rental Dataset."""
    
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Prepare the training data
        X_train = train.X  # Features
        y_train = train.y   # Target variable

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        model = Ridge()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)

        # Best model
        best_model = grid_search.best_estimator_

        # Serialize the model and scaler.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((best_model, scaler), model_file)

        print(f"Best model parameters: {grid_search.best_params_}")

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model, scaler = pickle.load(model_file)

        # Generate `predictions` with the test set predictions.
        X_test = test.X  # Features from the test set
        X_test_scaled = scaler.transform(X_test)  # Scale the test set
        predictions = model.predict(X_test_scaled)

        return predictions

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    predictions = main(main_args)
    if predictions is not None:
        print(predictions)

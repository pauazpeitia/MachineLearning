#!/usr/bin/env python3

#3d41c24d-1e20-459e-ab2e-5f0e184f26aa  --  Jose Mataix Perez
#5ca5e08e-855f-41d0-9025-06918d611fd2  --  Antonio Trujillo Reino
#e463771c-c409-4c11-b74f-687823d73cc2  --  Pau Azpeitia

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
import sklearn.linear_model
import sklearn.compose
import sklearn.pipeline

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        categoricalidxs = []
        floatidxs = []

        for i in range(np.shape(train.data)[1]):
            if all(train.data[j,i].is_integer() for j in range(np.shape(train.data)[0])):
                categoricalidxs.append(i)
            else:
                floatidxs.append(i)
        
        OH_transformer = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        SS_transformer = sklearn.preprocessing.StandardScaler()

        Transformation = sklearn.compose.ColumnTransformer([('a', OH_transformer, categoricalidxs), ('b', SS_transformer, floatidxs)])
        PolF_transformer = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)   

        LR_transformer = sklearn.linear_model.LogisticRegression(max_iter = 1000)

        pipe = sklearn.pipeline.Pipeline([('OHSS', Transformation), ('polF', PolF_transformer),('LR', LR_transformer)])
        
        model = pipe.fit(train.data, train.target)

        #Accuracy
        prediction = model.predict(train.data)

        train_hits = 0
        for i in range(np.shape(train.data)[0]):
            if(prediction[i] == train.target[i]):
                train_hits += 1

        accuracy = np.divide(train_hits, np.shape(train.data)[0])
        
        print("Accuracy: "+str(accuracy*100)+" %")

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        data = test.data
        
        predictions = model.predict(data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

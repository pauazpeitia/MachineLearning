#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
from sklearn.neural_network import MLPClassifier  # Agregar esta línea

parser = argparse.ArgumentParser()
# Estos argumentos serán configurados por ReCodEx
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset."""
    def __init__(self, name="mnist.train.npz", data_size=None, url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def main(args: argparse.Namespace) -> Optional[np.ndarray]:
    if args.predict is None:
        # Entrenamiento del modelo
        np.random.seed(args.seed)
        train = Dataset()

        # Crear y entrenar el modelo
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, random_state=args.seed)
        model.fit(train.data, train.target)

        # Guardar el modelo
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Cargar el modelo y hacer predicciones
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Hacer predicciones sobre el conjunto de prueba
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

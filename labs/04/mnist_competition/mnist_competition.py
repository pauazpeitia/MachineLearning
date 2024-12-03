#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.pipeline
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self, name="fiction-train.txt", url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read().splitlines()
        self.data = [sentence.translate(self.DIA_TO_NODIA) for sentence in self.target]

def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # Training phase
        np.random.seed(args.seed)
        train = Dataset()

        # Preparación de características y etiquetas
        X_train = []
        y_train = []

        for original, no_diacritics in zip(train.target, train.data):
            original_words = original.split(" ")
            no_diacritics_words = no_diacritics.split(" ")
            X_train.extend(no_diacritics_words)
            y_train.extend(original_words)

        # Características de n-gramas de caracteres
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 4))
        X_train = vectorizer.fit_transform(X_train)

        # Inicialización del modelo MLP
        mlp_model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=args.seed)

        # Pipeline de normalización y modelo MLP
        pipeline = sklearn.pipeline.Pipeline([
            ('scaler', sklearn.preprocessing.StandardScaler(with_mean=False)),
            ('mlp', mlp_model)
        ])

        # Búsqueda de hiperparámetros
        param_grid = {
            'mlp__alpha': [0.0001, 0.001, 0.01],
            'mlp__learning_rate': ['constant', 'adaptive'],
            'mlp__solver': ['adam', 'sgd'],
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Guardar el mejor modelo
        best_model = grid_search.best_estimator_
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((best_model, vectorizer), model_file)

    else:
        # Prediction phase
        if not os.path.exists(args.model_path):
            print("Error: el archivo del modelo no existe. Entrena el modelo antes de predecir.", file=sys.stderr)
            sys.exit(1)  # Salir si el modelo no existe

        test = Dataset(args.predict)

        # Cargar el modelo y el vectorizador
        with lzma.open(args.model_path, "rb") as model_file:
            best_model, vectorizer = pickle.load(model_file)

        # Transformar los datos de prueba y predecir
        X_test = vectorizer.transform(test.data)
        predictions = best_model.predict(X_test)

        # Formatear predicciones
        return "\n".join(predictions)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

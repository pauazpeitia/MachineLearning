import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())
    NODIA_TO_DIA = {nodia: dia for nodia, dia in zip(LETTERS_NODIA, LETTERS_DIA)}

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print(f"Downloading dataset {name}...", file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

    def generate_features_and_labels(self):
        features, labels = [], []
        for nodia_word, dia_word in zip(self.data.split(), self.target.split()):
            for i, char in enumerate(nodia_word):
                context = (
                    (nodia_word[i - 2] if i > 1 else "#") +
                    (nodia_word[i - 1] if i > 0 else "#") +
                    char +
                    (nodia_word[i + 1] if i < len(nodia_word) - 1 else "#") +
                    (nodia_word[i + 2] if i < len(nodia_word) - 2 else "#")
                )
                features.append(context)
                labels.append(dia_word[i])
        return features, labels


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        features, labels = train.generate_features_and_labels()

        model = Pipeline([
            ("vectorizer", TfidfVectorizer(analyzer="char", ngram_range=(1, 5))),
            ("classifier", MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=500))
        ])
        model.fit(features, labels)

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        predictions = []
        for word in test.data.split():
            word_prediction = ""
            for i, char in enumerate(word):
                context = (
                    (word[i - 2] if i > 1 else "#") +
                    (word[i - 1] if i > 0 else "#") +
                    char +
                    (word[i + 1] if i < len(word) - 1 else "#") +
                    (word[i + 2] if i < len(word) - 2 else "#")
                )
                word_prediction += model.predict([context])[0]
            predictions.append(word_prediction)

        return " ".join(predictions)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    result = main(main_args)
    if result:
        print(result)

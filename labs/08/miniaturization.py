import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import numpy as np
import sklearn.neural_network
from sklearn.preprocessing import StandardScaler, OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="miniaturization.model", type=str, help="Model path")

class Dataset:
    """MNIST Dataset."""
    def __init__(self, name="mnist.train.npz", data_size=None, url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)

class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"
        def fit(self, y):
            return self
        def transform(self, y):
            return y
        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(X, y, multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = y.shape[1]  # This now works because y is one-hot encoded
        return X, y

def main(args: argparse.Namespace) -> Optional[np.ndarray]:
    if args.predict is None:
        # Training phase
        np.random.seed(args.seed)
        train = Dataset()

        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train.data)
        y_train = train.target

        # One-hot encode the labels
        encoder = OneHotEncoder()  # No need for sparse=False
        y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()  # Convert to dense array

        # Create a small MLP with only a couple of layers and neurons to minimize model size
        model = MLPFullDistributionClassifier(
            hidden_layer_sizes=(512, 256),
            max_iter=1000, 
            random_state=args.seed
        )
        model.fit(X_train, y_train_one_hot)

        # Quantize model coefficients to reduce size
        model._optimizer = None  # Remove optimizer to save space
        for i in range(len(model.coefs_)):
            model.coefs_[i] = model.coefs_[i].astype(np.float16)  # Quantize coefficients to float16
        for i in range(len(model.intercepts_)):
            model.intercepts_[i] = model.intercepts_[i].astype(np.float16)  # Quantize intercepts

        # Serialize and compress the model
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Prediction phase
        test = Dataset(args.predict)
        X_test = StandardScaler().fit_transform(test.data)  # Normalize test data

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate predictions
        predictions = model.predict(X_test)
        return predictions

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
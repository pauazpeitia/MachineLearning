#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    # Load the dataset
    dataset = getattr(sklearn.datasets, f"load_{args.dataset}")()

    # Split the dataset into a train set and a test set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed
    )

    # Identify categorical (integer) and numerical columns
    categorical_columns = np.array([np.issubdtype(X_train[:, i].dtype, np.integer) for i in range(X_train.shape[1])])
    numerical_columns = ~categorical_columns

    # Define the transformer for categorical columns: OneHotEncoder
    categorical_transformer = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Define the transformer for numerical columns: StandardScaler
    numerical_transformer = sklearn.preprocessing.StandardScaler()

    # Combine them into a ColumnTransformer
    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numerical_transformer, numerical_columns)
        ]
    )

    # Add polynomial features of degree 2 to the preprocessed data
    polynomial_features = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)

    # Combine preprocessing and polynomial features into a pipeline
    pipeline = sklearn.pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ("poly", polynomial_features)
    ])

    # Fit the pipeline on the training data and transform both training and test data
    train_data = pipeline.fit_transform(X_train)
    test_data = pipeline.transform(X_test)

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(main_args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))),
                  *["..."] if dataset.shape[1] > 140 else [])

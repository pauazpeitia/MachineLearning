#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    # Load the dataset
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()
    X, y = dataset.data, dataset.target
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # For `linnerud`, we assume all features are numerical
    # We won't add any categorical transformation here since all are numerical
    numerical_transformer = sklearn.preprocessing.StandardScaler()

    # Create a ColumnTransformer for numerical features
    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, np.arange(X_train.shape[1]))  # all features are numerical
        ]
    )

    # Create a pipeline to include polynomial features
    pipeline = sklearn.pipeline.Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False))
    ])

    # Fit the pipeline on the training data and transform both train and test data
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

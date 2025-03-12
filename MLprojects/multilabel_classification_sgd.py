#!/usr/bin/env python3

#3d41c24d-1e20-459e-ab2e-5f0e184f26aa -- Jose Mataix Perez
#5ca5e08e-855f-41d0-9025-06918d611fd2 -- Antonio Trujillo Reino
#e463771c-c409-4c11-b74f-687823d73cc2 -- Pau Azpeitia Bergos

# Multilabel classification using Stochastic Gradient Descent (SGD).
# Predicts multiple labels per instance using sigmoid activation.
# Evaluates performance using micro and macro F1 scores.

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    generator = np.random.RandomState(args.seed)

    
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    #(n-hot encoding)
    target = np.zeros((args.data_size, args.classes), dtype=int)
    for i, labels in enumerate(target_list):
        target[i, labels] = 1

    # (bias)
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Dividir
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)


    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        for batch_start in range(0, train_data.shape[0], args.batch_size):
            batch_indices = permutation[batch_start:batch_start + args.batch_size]
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices]

            
            logits = batch_data @ weights
            probs = 1 / (1 + np.exp(-logits))  # Sigmoide para cada clase

            
            gradient = batch_data.T @ (probs - batch_target) / args.batch_size
            weights -= args.learning_rate * gradient

        
        def f1_score(y_true, y_pred, average='micro'):
            tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
            fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
            fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)

            if average == 'micro':
                tp = np.sum(tp)
                fp = np.sum(fp)
                fn = np.sum(fn)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return np.mean(f1) if average == 'macro' else f1

        
        train_preds = (train_data @ weights > 0).astype(int)
        test_preds = (test_data @ weights > 0).astype(int)

        
        train_f1_micro = f1_score(train_target, train_preds, average='micro')
        train_f1_macro = f1_score(train_target, train_preds, average='macro')
        test_f1_micro = f1_score(test_target, test_preds, average='micro')
        test_f1_macro = f1_score(test_target, test_preds, average='macro')

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")

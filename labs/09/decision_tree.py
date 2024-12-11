import argparse
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from collections import Counter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def gini_impurity(groups, classes):  #Recibe dos grupos y calcula el gini
    total_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups: #Realmente son dos grupos
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        group_classes = [row[-1] for row in group]  #Lista con las etiquetas (clase)
        for class_val in classes:
            proportion = group_classes.count(class_val) / size
            score += proportion ** 2
        gini += (1 - score) * (size / total_instances)
    return gini

def entropy(groups, classes):
    total_instances = sum(len(group) for group in groups)
    ent = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        group_classes = [row[-1] for row in group]
        scores = []
        for class_val in classes:
            proportion = group_classes.count(class_val) / size
            if proportion > 0:
                scores.append(-proportion * np.log2(proportion))
        ent += sum(scores) * (size / total_instances)
    return ent

def split_data(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

class DecisionTreeNode:
    def __init__(self):
        self.index = None
        self.value = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.prediction = None

class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_to_split=2, max_leaves=None):
        self.criterion = gini_impurity if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.max_leaves = max_leaves
        self.root = None
        self.leaf_count = 0  # Para llevar la cuenta de las hojas

    def fit(self, X, y):
        dataset = np.column_stack((X, y))   #Datos de entrenamiento y sus respectivos targets
        classes = np.unique(y)
        self.root = self._build_tree(dataset, classes, 0)

    def _build_tree(self, dataset, classes, depth):
        node = DecisionTreeNode()
        target_values = [row[-1] for row in dataset]

        #Parar si alguna condicion lo impide 
        if (self.max_depth is not None and depth >= self.max_depth) or len(set(target_values)) == 1 or len(dataset) < self.min_to_split or (self.max_leaves is not None and self.leaf_count >= self.max_leaves):
            node.is_leaf = True
            node.prediction = Counter(target_values).most_common(1)[0][0]
            self.leaf_count += 1 
            print(f"Leaf created. Total leaves: {self.leaf_count}") 
            return node #termina
        
        #mjr split
        best_index, best_value, best_score, best_groups = None, None, float('inf'), None
        for index in range(len(dataset[0]) - 1):
            unique_values = np.unique(dataset[:, index])
            split_points = (unique_values[:-1] + unique_values[1:]) / 2

            for value in split_points:
                groups = split_data(index, value, dataset)
                score = self.criterion(groups, classes)   #Aqui usa gini o entropy 

                if score < best_score:
                    best_index, best_value, best_score, best_groups = index, value, score, groups

        if best_score == float('inf') or best_groups is None:
            node.is_leaf = True
            print(f"Leaf created. Total leaves: {self.leaf_count}")
            node.prediction = Counter(target_values).most_common(1)[0][0]
            self.leaf_count += 1  
            return node

        left_data, right_data = best_groups
        node.index = best_index
        node.value = best_value

        node.left = self._build_tree(np.array(left_data), classes, depth + 1)
        node.right = self._build_tree(np.array(right_data), classes, depth + 1)

        return node

    def predict(self, X):
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, row, node):
        if node.is_leaf:
            return node.prediction
        if row[node.index] < node.value:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    
    # TODO: Manually create a decision tree on the training data.
    tree = DecisionTree(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_to_split=args.min_to_split,
        max_leaves= args.max_leaves
    )
    tree.fit(train_data, train_target) #main

    train_predictions = tree.predict(train_data)
    test_predictions = tree.predict(test_data)

    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_predictions)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    return 100 * train_accuracy, 100 * test_accuracy

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
#!/usr/bin/env python3

#3d41c24d-1e20-459e-ab2e-5f0e184f26aa  --  Jose Mataix Perez
#5ca5e08e-855f-41d0-9025-06918d611fd2  --  Antonio Trujillo Reino
#e463771c-c409-4c11-b74f-687823d73cc2  --  Pau Azpeitia

# Random Forest for classification tasks.
# Trains multiple decision trees with feature subsampling and bagging.
# Predicts classes using majority voting from the ensemble.

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1.0, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=44, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return np.sort(generator_feature_subsampling.choice(
            number_of_features, size=int(args.feature_subsampling * number_of_features), replace=False))

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)
    
    def criterion_value(groups):    
        value = 0.0
        for group in groups:
            if len(group) == 0:
                value.append(0.0)
                continue
            p = np.bincount(group[:, -1].astype(int)).astype(float)
            p /= len(group)
            p = p[p != 0]
            value -= len(group) * np.sum(p * np.log(p))
        return value

    def test_split(index, value, data): 
        left, right = [],[]
        left = data[data[:,index] < value]
        right = data[data[:,index] > value]
        return left, right

    def get_split(data):
        b_index, b_value, b_score, b_groups = 0, 0, float('+inf'), None
        number_of_features = len(data[0]) - 1
        sub_feat = subsample_features(number_of_features)

        for index in sub_feat:
            ts = np.unique(data[:, index])
            for i in range(len(ts) - 1):
                t = (ts[i] + ts[i + 1]) / 2
                groups = test_split(index, t, data)
                crit_value = criterion_value(groups)

                if crit_value < b_score:
                    b_index, b_value, b_score, b_groups = index, t, crit_value, groups
        return {'index': b_index, 'value': b_value, 'score': b_score, 'groups': b_groups}

    def to_terminal(group):
        targets = group[:,-1].astype(int) 
        return np.argmax(np.bincount(targets))
    
    def split(node, depth, args):
        max_depth = args.max_depth
        left, right = node['groups']
        del(node['groups'])

        if len(left)==0 or len(right)==0: 
            node = to_terminal(np.vstack((left,right)))
            return

        if max_depth != None:
            if depth >= max_depth:
                node['left'], node['right'] = to_terminal(left), to_terminal(right)
                return

        if not np.all(left[:,-1] == left[0][-1]):
            node['left'] = get_split(left)
            split(node['left'], depth+1, args)
        else:
            node['left'] = to_terminal(left)
        
        if not np.all(right[:,-1] == right[0][-1]):
            node['right'] = get_split(right)
            split(node['right'], depth+1, args)
        else:
            node['right'] = to_terminal(right)
    
    def predict(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']
    
    def build_tree(train, args):
        groups = train,[]
        root = get_split(train)
        split(root, int(1), args)
        
        return root
    train = np.c_[train_data,train_target]  
    k = 0 
    trees = []

    while k < args.trees:
        
        if  args.bagging == True:
            dataset_indices = bootstrap_dataset(train_data)
            tree = build_tree(train[dataset_indices],args)
        else: 
            tree = build_tree(train,args)
            
        trees.append(tree)
    
        k += 1


    pred_train = []
    for row in train_data:
        pred_row = []
        for tree in trees:
            pred_row.append(predict(tree,row))
        pred_train.append(np.argmax(np.bincount(pred_row)))
    
    pred_test = []
    for row in test_data:
        pred_row = []
        for tree in trees:
            pred_row.append(predict(tree,row))
        pred_test.append(np.argmax(np.bincount(pred_row)))

    train_accuracy = sklearn.metrics.accuracy_score(train_target,pred_train)
    test_accuracy = sklearn.metrics.accuracy_score(test_target,pred_test)

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))

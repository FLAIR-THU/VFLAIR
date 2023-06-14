import random
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from evaluates.MainTaskTVFL import MainTaskTVFL
from load.LoadTreeParty import load_tree_parties


def evaluate_performance(tvfl, X_train, y_train, X_test, y_test):
    y_pred_train = tvfl.clf.predict_proba(X_train)
    y_pred_test = tvfl.clf.predict_proba(X_test)
    train_auc = roc_auc_score(y_train, np.array(y_pred_train)[:, 1])
    test_auc = roc_auc_score(y_test, np.array(y_pred_test)[:, 1])
    print(f" train auc: {train_auc}, test auc: {test_auc}")


class Args:
    def __init__(self, datasets, featureid_lists, y):
        self.model_type = "xgboost"

        self.seed_base = seed
        self.number_of_trees = 3
        self.depth = 2
        self.min_leaf = 1
        self.subsample_cols = 0.8
        self.max_bin = 4
        self.use_missing_value = False

        self.k = 2
        self.datasets = datasets
        self.y = y
        self.featureid_lists = featureid_lists

        self.use_encryption = False
        self.key_length = 128


if __name__ == "__main__":
    seed = 42
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )

    args = Args(
        [
            X_train[:, : int(X_train.shape[1] / 2)],
            X_train[:, int(X_train.shape[1] / 2) :],
        ],
        [
            range(int(X_train.shape[1] / 2)),
            range(int(X_train.shape[1] / 2), X_train.shape[1]),
        ],
        y_train,
    )

    for model_type in ["xgboost", "randomforest"]:
        for use_encryption in [False, True]:
            print(f"type of model: {model_type}, encryption:{use_encryption}")
            args.model_type = model_type
            args.use_encryption = use_encryption
            args = load_tree_parties(args)

            tvfl = MainTaskTVFL(args)

            start = time.time()
            tvfl.train()
            end = time.time()

            print(f" training time: {end - start} [s]")
            evaluate_performance(tvfl, X_train, y_train, X_test, y_test)

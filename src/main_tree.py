import random
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from evaluates.MainTaskTVFL import MainTaskTVFL
from load.LoadTreeParty import load_tree_parties
from load.LoadTreeConfigs import load_tree_configs

import argparse


def evaluate_performance(tvfl, X_train, y_train, X_test, y_test):
    y_pred_train = tvfl.clf.predict_proba(X_train)
    y_pred_test = tvfl.clf.predict_proba(X_test)
    train_auc = roc_auc_score(y_train, np.array(y_pred_train)[:, 1])
    test_auc = roc_auc_score(y_test, np.array(y_pred_test)[:, 1])
    print(f" train auc: {train_auc}, test auc: {test_auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("tree")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--configs",
        type=str,
        default="basic_configs_tree",
        help="configure json file path",
    )
    args = parser.parse_args()
    args = load_tree_configs(args.configs, args)

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=args.seed
    )
    print(X_train.shape)
    datasets = [
        X_train[:, : int(X_train.shape[1] / 30)],
        X_train[:, int(X_train.shape[1] / 30) :],
    ]
    featureid_lists = [
        range(int(X_train.shape[1] / 30)),
        range(int(X_train.shape[1] / 30), X_train.shape[1]),
    ]
    args.datasets = datasets
    args.y = y_train
    args.featureid_lists = featureid_lists

    print(f"type of model: {args.model_type}, encryption:{args.use_encryption}")
    args = load_tree_parties(args)

    tvfl = MainTaskTVFL(args)

    start = time.time()
    tvfl.train()
    end = time.time()

    print(f" training time: {end - start} [s]")
    evaluate_performance(tvfl, X_train, y_train, X_test, y_test)

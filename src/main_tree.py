import random
import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from evaluates.MainTaskTVFL import MainTaskTVFL
from load.LoadTreeParty import load_tree_parties
from load.LoadTreeConfigs import load_tree_configs

import argparse


def evaluate_performance(tvfl, X_train, y_train, X_test, y_test, grid=False):
    y_pred_train = tvfl.clf.predict_proba(X_train)
    y_pred_test = tvfl.clf.predict_proba(X_test)
    if grid:
        thresholds = np.arange(0, 1.01, 0.01)
        best_threshold = 0
        best_accuracy = 0

        for threshold in thresholds:
            y_pred_binary = (np.array(y_pred_train)[:, 1] >= threshold).astype(int)
            accuracy = accuracy_score(y_train, y_pred_binary)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        train_acc = best_accuracy
        test_acc = accuracy_score(
            y_test, (np.array(y_pred_test)[:, 1] >= best_threshold).astype(int)
        )
    else:
        train_acc = accuracy_score(y_train, np.argmax(y_pred_train, axis=1))
        test_acc = accuracy_score(y_test, np.argmax(y_pred_test, axis=1))

    print(f" train acc: {train_acc}, test acc: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("tree")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--dataset", type=str, default="breastcancer")
    parser.add_argument("--bin", type=int, default=None)
    parser.add_argument(
        "--configs",
        type=str,
        default="basic_configs_tree",
        help="configure json file path",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
    )
    args = parser.parse_args()
    args = load_tree_configs(args.configs, args)

    random.seed(args.seed)

    if args.dataset == "credit":
        df = pd.read_csv(os.path.join("tabledata", "UCI_Credit_Card.csv"))

        X = df[
            [
                "LIMIT_BAL",
                "SEX",
                "EDUCATION",
                "MARRIAGE",
                "AGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
            ]
        ].values
        y = df["default.payment.next.month"].values
        featureid_lists = [
            range(int(X.shape[1] / 2)),
            range(int(X.shape[1] / 2), X.shape[1]),
        ]

    elif args.dataset == "nursery":
        df = pd.read_csv(os.path.join("tabledata", "nursery.data"), header=None)
        df[8] = LabelEncoder().fit_transform(df[8].values)
        X_d = df.drop(8, axis=1)
        X_a = pd.get_dummies(
            X_d[X_d.columns[: int(len(X_d.columns) / 2)]], drop_first=True, dtype=int
        )
        X_p = pd.get_dummies(
            X_d[X_d.columns[int(len(X_d.columns) / 2):]], drop_first=True, dtype=int
        )
        featureid_lists = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df[8].values

    elif args.dataset == "digit":
        data = load_digits()
        X = data.data
        y = data.target
        featureid_lists = [
            range(int(X.shape[1] / 2)),
            range(int(X.shape[1] / 2), X.shape[1]),
        ]

    else:
        data = load_breast_cancer()
        X = data.data
        y = data.target
        featureid_lists = [
            range(int(X.shape[1] / 2)),
            range(int(X.shape[1] / 2), X.shape[1]),
        ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(X_train.shape)
    datasets = [
        X_train[:, featureid_lists[0]],
        X_train[:, featureid_lists[1]],
    ]
    args.datasets = datasets
    args.y = y_train
    args.featureid_lists = featureid_lists

    if args.bin is not None:
        args.max_bin = args.bin
    
    print(f"type of model: {args.model_type}, encryption:{args.use_encryption}")
    args = load_tree_parties(args)


    tvfl = MainTaskTVFL(args)

    start = time.time()
    tvfl.train()
    end = time.time()

    print(f" training time: {end - start} [s]")
    evaluate_performance(tvfl, X_train, y_train, X_test, y_test, args.grid)

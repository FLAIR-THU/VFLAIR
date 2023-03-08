import argparse

import numpy as np
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from evaluates.attacks.ID2Graph import ID2Graph
from models.tree import RandomForestClassifier, XGBoostClassifier
from party.tree import RandomForestParty, XGBoostParty

use_missing_value = False


def baseline(X_train, y_train, num_classes, seed):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_normalized = min_max_scaler.fit_transform(X_train[:, int(X_train.shape[1] / 2) :])
    kmeans = KMeans(n_clusters=num_classes, random_state=seed).fit(X_normalized)
    baseline_v = metrics.v_measure_score(y_train, kmeans.labels_)
    return baseline_v


def experiment(
    model_type,
    eta,
    min_leaf,
    subsample_cols,
    max_bin,
    depth,
    number_of_trees,
    seed_base,
):
    data = load_breast_cancer()
    X = data.data
    y = data.target
    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed_base
    )

    baseline_v = baseline(X_train, y_train, num_classes, seed_base)

    if model_type == "xgboost":
        party_0 = XGBoostParty(
            X_train[:, : int(X_train.shape[1] / 2)],
            num_classes,
            range(int(X_train.shape[1] / 2)),
            0,
            min_leaf,
            subsample_cols,
            max_bin,
            use_missing_value,
        )
        party_1 = XGBoostParty(
            X_train[:, int(X_train.shape[1] / 2) :],
            num_classes,
            range(int(X_train.shape[1] / 2), X_train.shape[1]),
            1,
            min_leaf,
            subsample_cols,
            max_bin,
            use_missing_value,
        )
        clf = XGBoostClassifier(
            num_classes, boosting_rounds=number_of_trees, depth=depth, active_party_id=0
        )
    elif model_type == "randomforest":
        party_0 = RandomForestParty(
            X_train[:, : int(X_train.shape[1] / 2)],
            num_classes,
            range(int(X_train.shape[1] / 2)),
            0,
            min_leaf,
            subsample_cols,
        )
        party_1 = RandomForestParty(
            X_train[:, int(X_train.shape[1] / 2) :],
            num_classes,
            range(int(X_train.shape[1] / 2), X_train.shape[1]),
            1,
            min_leaf,
            subsample_cols,
        )
        clf = RandomForestClassifier(
            num_classes, num_trees=number_of_trees, depth=depth, active_party_id=0
        )
    else:
        raise ValueError("model type should be `xgboost` or `randomforest`")

    parties = [party_0, party_1]
    clf.fit(parties, y_train)

    y_pred_test = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, np.array(y_pred_test)[:, 1])

    class AttackArgs:
        attack_name = "ID2Graph"
        attack_configs = {"party": party_1, "eta": eta}
        seed = seed_base

    attacker = ID2Graph(clf, AttackArgs)
    result = attacker.attack()
    id2graph_v = metrics.v_measure_score(y_train, result)

    return id2graph_v, baseline_v, test_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("id2graph")
    parser.add_argument(
        "--model", type=str, default="xgboost", help="attack xgboost or randomforest"
    )
    parser.add_argument("--num_trials", type=int, default=5, help="number of trials")
    parser.add_argument("--eta", type=float, default=0.6, help="discount factor")
    parser.add_argument(
        "--min_leaf",
        type=int,
        default=1,
        help="minmum number of samples within a leaf",
    )
    parser.add_argument(
        "--subsample_cols",
        type=float,
        default=0.8,
        help="subsampling ratio of features used for training each tree",
    )
    parser.add_argument(
        "--max_bin",
        type=int,
        default=4,
        help="maximum number of bins (used only for xgboost)",
    )
    parser.add_argument("--depth", type=int, default=6, help="maximum depth")
    parser.add_argument(
        "--number_of_trees", type=int, default=5, help="number of trees"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    result = []
    for s in range(
        args.seed,
        args.seed + args.num_trials,
    ):
        result.append(
            experiment(
                args.model,
                args.eta,
                args.min_leaf,
                args.subsample_cols,
                args.max_bin,
                args.depth,
                args.number_of_trees,
                s,
            )
        )

    averaged_result = np.array(result).mean(axis=0)
    if averaged_result[0] > averaged_result[1]:
        print("Attack Success")
    else:
        print("Attack Failure")
    print(f"AUC on Test Data: {averaged_result[2]}")
    print(f"V-measure of ID2Graph: {averaged_result[0]}")
    print(f"V-measure of BaseLine: {averaged_result[1]}")

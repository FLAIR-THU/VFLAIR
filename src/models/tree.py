import os
import sys

sys.path.append(os.pardir)

from typing import Callable, List

import numpy as np

from .tree_loss import BCELoss, CELoss, sigmoid, softmax
from .tree_node_core import Tree
from .tree_node_xgboost import XGBoostNode


class XGBoostTree(Tree):
    def fit(
        self,
        parties: List,
        y: List[float],
        num_classes: int,
        gradient: List[List[float]],
        hessian: List[List[float]],
        min_child_weight: float,
        lam: float,
        gamma: float,
        eps: float,
        depth: int,
        active_party_id: int = -1,
        use_only_active_party: bool = False,
        n_job: int = 1,
        custom_secure_cond_func: Callable = (lambda _: False),
    ):
        idxs = list(range(len(y)))
        for i in range(len(parties)):
            parties[i].subsample_columns()
        self.num_row = len(y)
        self.dtree = XGBoostNode(
            parties,
            y,
            num_classes,
            gradient,
            hessian,
            idxs,
            min_child_weight,
            lam,
            gamma,
            eps,
            depth,
            active_party_id,
            use_only_active_party,
            n_job,
            custom_secure_cond_func,
        )

    def free_intermediate_resources(self):
        self.dtree.y = []


class XGBoostBase:
    def __init__(
        self,
        num_classes: int,
        subsample_cols: float = 0.8,
        min_child_weight: float = -np.inf,
        depth: int = 5,
        learning_rate: float = 0.4,
        boosting_rounds: int = 5,
        lam: float = 1.5,
        gamma: float = 1,
        eps: float = 0.1,
        active_party_id: int = -1,
        completelly_secure_round: int = 0,
        init_value: float = 1.0,
        n_job: int = 1,
        custom_secure_cond_func: Callable = (lambda _: False),
        save_loss: bool = True,
    ):
        self.num_classes = num_classes
        self.subsample_cols = subsample_cols
        self.min_child_weight = min_child_weight
        self.depth = depth
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lam = lam
        self.gamma = gamma
        self.eps = eps
        self.active_party_id = active_party_id
        self.completelly_secure_round = completelly_secure_round
        self.init_value = init_value
        self.n_job = n_job
        self.custom_secure_cond_func = custom_secure_cond_func
        self.save_loss = save_loss
        if num_classes == 2:
            self.lossfunc_obj = BCELoss()
        else:
            self.lossfunc_obj = CELoss(num_classes)
        self.init_pred = None
        self.estimators = []
        self.logging_loss = []

    def get_init_pred(self, y: np.ndarray) -> List[List[float]]:
        pass

    def load_estimators(self, _estimators: List) -> None:
        self.estimators = _estimators

    def clear(self) -> None:
        self.estimators.clear()
        self.logging_loss.clear()

    def get_estimators(self) -> List:
        return self.estimators

    def fit(self, parties: List, y: np.ndarray) -> None:
        row_count = len(y)
        base_pred = []
        if not self.estimators:
            self.init_pred = self.get_init_pred(y)
            base_pred = self.init_pred.copy()
        else:
            base_pred = np.zeros((row_count, self.num_classes))
            for i in range(len(self.estimators)):
                pred_temp = self.estimators[i].get_train_prediction()
                base_pred += self.learning_rate * np.array(pred_temp)
        for i in range(self.boosting_rounds):
            grad, hess = self.lossfunc_obj.get_grad(
                base_pred, y
            ), self.lossfunc_obj.get_hess(base_pred, y)
            boosting_tree = XGBoostTree()
            boosting_tree.fit(
                parties,
                y,
                self.num_classes,
                grad,
                hess,
                self.min_child_weight,
                self.lam,
                self.gamma,
                self.eps,
                self.depth,
                self.active_party_id,
                (self.completelly_secure_round > i),
                self.n_job,
                self.custom_secure_cond_func,
            )
            pred_temp = boosting_tree.get_train_prediction()
            base_pred += self.learning_rate * np.array(pred_temp)

            self.estimators.append(boosting_tree)

            if self.save_loss:
                self.logging_loss.append(self.lossfunc_obj.get_loss(base_pred, y))

    def predict_raw(self, X):
        pred_dim = 1 if self.num_classes == 2 else self.num_classes
        row_count = len(X)
        y_pred = [[self.init_value] * pred_dim for _ in range(row_count)]
        estimators_num = len(self.estimators)
        for i in range(estimators_num):
            y_pred_temp = self.estimators[i].predict(X)
            for j in range(row_count):
                for c in range(pred_dim):
                    y_pred[j][c] += self.learning_rate * y_pred_temp[j][c]
        return y_pred

    def free_intermediate_resources(self):
        estimators_num = len(self.estimators)
        for i in range(estimators_num):
            self.estimators[i].free_intermediate_resources()


class XGBoostClassifier(XGBoostBase):
    def get_init_pred(self, y):
        init_pred = np.full((len(y), self.num_classes), self.init_value, dtype=float)
        return init_pred.tolist()

    def predict_proba(self, x):
        raw_score = self.predict_raw(x)
        row_count = len(x)
        predicted_probas = np.zeros((row_count, self.num_classes), dtype=float)
        for i in range(row_count):
            if self.num_classes == 2:
                predicted_probas[i][1] = sigmoid(raw_score[i][0])
                predicted_probas[i][0] = 1 - predicted_probas[i][1]
            else:
                predicted_probas[i] = softmax(raw_score[i])
        return predicted_probas.tolist()

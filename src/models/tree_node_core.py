import math
from abc import abstractmethod
from typing import List, Tuple

import numpy as np


class Node:
    def __init__(self):
        self.parties = None
        self.idxs = []
        self.num_classes = 0
        self.depth = 0
        self.active_party_id = 0
        self.n_job = 0
        self.party_id = 0
        self.record_id = 0
        self.row_count = 0
        self.num_parties = 0
        self.score = 0.0
        self.val = []
        self.best_party_id = -1
        self.best_col_id = -1
        self.best_threshold_id = -1
        self.best_score = -1 * np.inf
        self.is_leaf_flag = -1
        self.is_pure_flag = -1
        self.secure_flag_exclude_passive_parties = False
        self.not_splitted_flag = False

    @abstractmethod
    def get_idxs(self) -> List[int]:
        pass

    @abstractmethod
    def get_party_id(self) -> int:
        pass

    @abstractmethod
    def get_record_id(self) -> int:
        pass

    @abstractmethod
    def get_val(self) -> List[float]:
        pass

    @abstractmethod
    def get_score(self) -> float:
        pass

    @abstractmethod
    def get_num_parties(self) -> int:
        pass

    @abstractmethod
    def compute_weight(self) -> List[float]:
        pass

    @abstractmethod
    def find_split(self) -> Tuple[int, int, int]:
        pass

    @abstractmethod
    def make_children_nodes(
            self, best_party_id: int, best_col_id: int, best_threshold_id: int
    ):
        pass

    @abstractmethod
    def is_leaf(self) -> bool:
        pass

    @abstractmethod
    def is_pure(self) -> bool:
        pass


class NodeAPI:
    def __init__(self) -> None:
        pass

    def print(self, node: Node, target_party_id: int = -1) -> str:
        result, skip_flag = self.recursive_print(node, "", False, target_party_id)
        if skip_flag:
            return ""
        else:
            return result

    def print_leaf(self, node: Node) -> str:
        node_info = str(node.get_val()[0])
        node_info += ", " + str(len(node.idxs))
        return node_info

    def recursive_print(
            self, node: Node, prefix: str, isleft: bool, target_party_id: int = -1
    ) -> Tuple[str, bool]:
        node_info = ""
        skip_flag = False
        if node.is_leaf_flag:
            skip_flag = (
                    node.depth <= 0
                    and target_party_id != -1
                    and node.party_id != target_party_id
            )
            if skip_flag:
                node_info = ""
            else:
                node_info = self.print_leaf(node)
            node_info = prefix + "|-- " + node_info
            node_info += "\n"
        else:
            node_info += str(node.get_party_id())
            node_info += ", "
            node_info += str(node.get_record_id())
            if node.secure_flag_exclude_passive_parties:
                node_info += " *"
            node_info = prefix + "|-- " + node_info

            next_prefix = ""
            if isleft:
                next_prefix += "|    "
            else:
                next_prefix += "     "

            left_node_info_and_skip_flag = self.recursive_print(
                node.left, prefix + next_prefix, True, target_party_id
            )
            right_node_info_and_skip_flag = self.recursive_print(
                node.right, prefix + next_prefix, False, target_party_id
            )
            if left_node_info_and_skip_flag[1] and right_node_info_and_skip_flag[1]:
                node_info += " -> " + self.print_leaf(node)
                node_info += "\n"
            else:
                node_info += "\n"
                node_info += left_node_info_and_skip_flag[0]
                node_info += right_node_info_and_skip_flag[0]

            skip_flag = False

        return node_info, skip_flag

    def predict_row(self, node: Node, xi: List[float]) -> List[float]:
        que = []
        que.append(node)
        while que:
            temp_node = que.pop(0)
            if temp_node.is_leaf_flag:
                return temp_node.val
            else:
                if node.parties[temp_node.party_id].is_left(temp_node.record_id, xi):
                    que.append(temp_node.left)
                else:
                    que.append(temp_node.right)
        nan_vec = [math.nan for _ in range(node.num_classes)]
        return nan_vec

    def predict(self, node: Node, x_new: List[List[float]]) -> List[List[float]]:
        x_new_size = len(x_new)
        y_pred = []
        for i in range(x_new_size):
            y_pred.append(self.predict_row(node, x_new[i]))
        return y_pred


class Tree:
    def __init__(self) -> None:
        self.dtree = None
        self.nodeapi = NodeAPI()
        self.num_row = None

    def get_root_node(self) -> Node:
        return self.dtree

    def predict(self, X: List[List[float]]) -> List[List[float]]:
        return self.nodeapi.predict(self.dtree, X)

    def extract_train_prediction_from_node(
            self, node: Node
    ) -> List[Tuple[List[int], List[List[float]]]]:
        if node.is_leaf_flag:
            result = [(node.idxs, [node.val] * len(node.idxs))]
            return result
        else:
            left_result = self.extract_train_prediction_from_node(node.left)
            right_result = self.extract_train_prediction_from_node(node.right)
            return left_result + right_result

    def get_train_prediction(self) -> List[List[float]]:
        result = self.extract_train_prediction_from_node(self.dtree)
        y_train_pred = [[] for _ in range(self.num_row)]
        for indices, values in result:
            for i, val in zip(indices, values):
                y_train_pred[i] = val
        return y_train_pred

    def print(self, target_party_id: int = -1) -> str:
        return self.nodeapi.print(self.dtree, target_party_id)

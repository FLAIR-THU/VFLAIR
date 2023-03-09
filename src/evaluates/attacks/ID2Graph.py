import os
import sys

sys.path.append(os.pardir)


from queue import Queue

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans

from evaluates.attacks.attacker import Attacker
from models.tree_node_core import Node


def travase_nodes_to_extract_adjacency_matrix(
    node: Node, adj_mat: np.ndarray, weight: float
) -> None:
    que = Queue()
    que.put(node)
    temp_node = None
    temp_idxs_size = None
    while not que.empty():
        temp_node = que.get()

        if temp_node.is_leaf_flag:
            if not temp_node.not_splitted_flag:
                temp_idxs_size = len(temp_node.idxs)
                for i in range(temp_idxs_size):
                    for j in range(i + 1, temp_idxs_size):
                        adj_mat[temp_node.idxs[i]][temp_node.idxs[j]] += weight
                        adj_mat[temp_node.idxs[j]][temp_node.idxs[i]] += weight

        else:
            not_splitted_flag = (
                temp_node.left.not_splitted_flag and temp_node.right.not_splitted_flag
            )
            secure_exclude_flag = (
                temp_node.left.secure_flag_exclude_passive_parties
                and temp_node.right.secure_flag_exclude_passive_parties
            )
            exclude_flag = not_splitted_flag or secure_exclude_flag

            if exclude_flag:
                temp_idxs_size = len(temp_node.idxs)
                for i in range(temp_idxs_size):
                    for j in range(i + 1, temp_idxs_size):
                        adj_mat[temp_node.idxs[i]][temp_node.idxs[j]] += weight
                        adj_mat[temp_node.idxs[j]][temp_node.idxs[i]] += weight

            if (
                not temp_node.left.secure_flag_exclude_passive_parties
                or not temp_node.right.secure_flag_exclude_passive_parties
            ):
                que.put(temp_node.left)
                que.put(temp_node.right)

        # temp_node.idxs.clear()
        # temp_node.idxs.shrink_to_fit()
        # temp_node.val.clear()
        # temp_node.val.shrink_to_fit()


def extract_adjacency_matrix_from_forest(model, skip_round=0, eta=0.3):
    num_row = model.estimators[0].num_row
    adj_matrix = np.zeros((num_row, num_row), dtype=float)

    for i, estimator in enumerate(model.estimators):
        if i >= skip_round:
            travase_nodes_to_extract_adjacency_matrix(
                estimator.dtree,
                adj_matrix,
                eta ** float(i - skip_round),
            )

    return adj_matrix


class ID2Graph(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.clf = top_vfl
        self.party = args.attack_configs["party"]  # parties that launch attacks
        self.eta = args.attack_configs["eta"]
        self.seed = args.seed

    def attack(self):
        adj_mat = extract_adjacency_matrix_from_forest(self.clf, eta=self.eta)

        G = nx.from_numpy_array(adj_mat)
        coms = nx_comm.louvain_communities(G, seed=self.seed)

        num_row = adj_mat.shape[0]
        x_com = np.zeros((num_row, len(coms)), dtype=int)
        for i, com in enumerate(coms):
            for idx in com:
                x_com[idx][i] = 1

        min_max_scaler = preprocessing.MinMaxScaler()
        X_normalized = min_max_scaler.fit_transform(self.party.x)
        X_train_with_com = np.concatenate([X_normalized, x_com], axis=1)

        kmeans_with_com = KMeans(
            n_clusters=self.clf.num_classes, random_state=self.seed
        ).fit(X_train_with_com)

        return kmeans_with_com.labels_

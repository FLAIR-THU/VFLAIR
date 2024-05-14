import math
import threading
from typing import List

from .tree_node_core import Node
from .idlmid import is_satisfied_mi_bound


def calc_giniimp(tot_cnt: float, class_cnt: List[float]) -> float:
    num_classes = len(class_cnt)
    giniimp = 1.0
    for c in range(num_classes):
        temp_class_ratio = class_cnt[c] / tot_cnt
        giniimp -= temp_class_ratio * temp_class_ratio
    return giniimp


class RandomForestNode(Node):
    def __init__(
            self,
            parties_,
            y_,
            y_onehot_encoded_,
            num_classes_,
            idxs_,
            depth_,
            prior_,
            mi_bound_=-1,
            active_party_id_=-1,
            use_only_active_party_=False,
            n_job_=1,
            y_onehot_encoded_encrypted=None,
    ):
        super().__init__()
        self.parties = parties_
        self.y = y_
        self.y_onehot_encoded = y_onehot_encoded_
        self.num_classes = num_classes_
        self.idxs = idxs_
        self.depth = depth_
        self.prior = prior_
        self.mi_bound = mi_bound_
        self.active_party_id = active_party_id_
        self.use_only_active_party = use_only_active_party_
        self.n_job = n_job_
        self.y_onehot_encoded_encrypted = y_onehot_encoded_encrypted

        self.left = None
        self.right = None
        self.giniimp = 0.0

        self.entire_datasetsize = 0.0
        self.entire_class_cnt = [0 for _ in range(num_classes_)]

        self.secure_flag_exclude_passive_parties = use_only_active_party_
        self.is_all_subsequent_children_contaminated = False

        self.row_count = len(idxs_)
        self.num_parties = len(parties_)

        for i in range(self.row_count):
            self.entire_class_cnt[int(self.y[idxs_[i]])] += 1.0
        self.giniimp = self.compute_giniimp()
        self.val = self.compute_weight()

        best_split = self.find_split()

        if self.is_leaf():
            self.is_leaf_flag = 1
        else:
            self.is_leaf_flag = 0

        if self.is_leaf_flag == 0:
            self.party_id = best_split[0]
            if self.party_id != -1:
                self.record_id = parties_[self.party_id].insert_lookup_table(
                    best_split[1], best_split[2]
                )
                self.make_children_nodes(best_split[0], best_split[1], best_split[2])
            else:
                self.is_leaf_flag = 1

    def get_idxs(self) -> List[int]:
        return self.idxs

    def get_party_id(self) -> int:
        return self.party_id

    def get_record_id(self) -> int:
        return self.record_id

    def get_val(self) -> List[float]:
        return self.val

    def get_score(self) -> float:
        return self.score

    def get_left(self) -> "RandomForestNode":
        return self.left

    def get_right(self) -> "RandomForestNode":
        return self.right

    def get_num_parties(self) -> int:
        return self.num_parties

    def get_num_parties_per_process(self, n_job, num_parties) -> List[int]:
        num_parties_per_process = [num_parties // n_job for _ in range(n_job)]
        for i in range(num_parties % n_job):
            num_parties_per_process[i] += 1
        return num_parties_per_process

    def compute_giniimp(self) -> float:
        temp_y_class_cnt = [0 for _ in range(self.num_classes)]
        for r in range(self.row_count):
            temp_y_class_cnt[int(self.y[self.idxs[r]])] += 1

        giniimp = 1
        for c in range(self.num_classes):
            temp_ratio_square = temp_y_class_cnt[c] / self.row_count
            giniimp -= temp_ratio_square * temp_ratio_square

        return giniimp

    def compute_weight(self) -> float:
        class_ratio = [0 for _ in range(self.num_classes)]
        for r in range(self.row_count):
            class_ratio[int(self.y[self.idxs[r]])] += 1 / float(self.row_count)
        return class_ratio

    def find_split_per_party(
            self, party_id_start, temp_num_parties, tot_cnt, temp_y_class_cnt
    ):
        temp_left_class_cnt = [0 for _ in range(self.num_classes)]
        temp_right_class_cnt = [0 for _ in range(self.num_classes)]

        for temp_party_id in range(party_id_start, party_id_start + temp_num_parties):
            if (
                    temp_party_id != self.active_party_id
                    and self.y_onehot_encoded_encrypted is not None
            ):
                search_results_encrypted = self.parties[
                    temp_party_id
                ].greedy_search_split(self.idxs, self.y_onehot_encoded_encrypted)
                search_results = []
                for j in range(len(search_results_encrypted)):
                    search_results.append([])
                    for k in range(len(search_results_encrypted[j])):
                        tls = search_results_encrypted[j][k][0]
                        tlc = self.parties[self.active_party_id].decrypt_1dlist(
                            search_results_encrypted[j][k][1]
                        )
                        search_results[-1].append((tls, tlc))
            else:
                search_results = self.parties[temp_party_id].greedy_search_split(
                    self.idxs, self.y_onehot_encoded
                )

            num_search_results = len(search_results)
            for j in range(num_search_results):
                temp_left_size = 0

                for c in range(self.num_classes):
                    temp_left_class_cnt[c] = 0
                    temp_right_class_cnt[c] = 0

                temp_num_search_results_j = len(search_results[j])
                for k in range(temp_num_search_results_j):
                    temp_left_size += search_results[j][k][0]
                    temp_right_size = tot_cnt - temp_left_size

                    for c in range(self.num_classes):
                        temp_left_class_cnt[c] += search_results[j][k][1][c]
                        temp_right_class_cnt[c] = (
                                temp_y_class_cnt[c] - temp_left_class_cnt[c]
                        )

                    if (temp_party_id != self.active_party_id) and (
                            (
                                    not is_satisfied_mi_bound(
                                        self.num_classes,
                                        self.mi_bound,
                                        temp_left_size,
                                        len(self.y),
                                        self.entire_class_cnt,
                                        self.prior,
                                        temp_left_class_cnt,
                                    )
                            )
                            or (
                                    not is_satisfied_mi_bound(
                                        self.num_classes,
                                        self.mi_bound,
                                        temp_right_size,
                                        len(self.y),
                                        self.entire_class_cnt,
                                        self.prior,
                                        temp_right_class_cnt,
                                    )
                            )
                    ):
                        continue

                    temp_left_giniimp = calc_giniimp(
                        temp_left_size, temp_left_class_cnt
                    )
                    temp_right_giniimp = calc_giniimp(
                        temp_right_size, temp_right_class_cnt
                    )
                    temp_giniimp = temp_left_giniimp * (
                            temp_left_size / tot_cnt
                    ) + temp_right_giniimp * (temp_right_size / tot_cnt)

                    temp_score = self.giniimp - temp_giniimp
                    if temp_score > self.best_score:
                        self.best_score = temp_score
                        self.best_party_id = temp_party_id
                        self.best_col_id = j
                        self.best_threshold_id = k

    def find_split(self):
        """
        Find the best split among all thresholds received from all clients.

        :return: Tuple[int, int, int]
        """
        self.temp_score = 0.0
        tot_cnt = self.row_count

        temp_y_class_cnt = [0 for _ in range(self.num_classes)]
        for r in range(self.row_count):
            temp_y_class_cnt[int(self.y[self.idxs[r]])] += 1

        if self.use_only_active_party:
            self.find_split_per_party(
                self.active_party_id, 1, tot_cnt, temp_y_class_cnt
            )
        else:
            if self.n_job == 1:
                self.find_split_per_party(
                    0, self.num_parties, tot_cnt, temp_y_class_cnt
                )
            else:
                num_parties_per_thread = self.get_num_parties_per_process(
                    self.n_job, self.num_parties
                )

                cnt_parties = 0
                threads_parties = []
                for i in range(self.n_job):
                    local_num_parties = num_parties_per_thread[i]
                    temp_th = threading.Thread(
                        target=lambda: self.find_split_per_party(
                            cnt_parties, local_num_parties, tot_cnt, temp_y_class_cnt
                        )
                    )
                    temp_th.start()
                    threads_parties.append(temp_th)
                    cnt_parties += num_parties_per_thread[i]
                for i in range(self.num_parties):
                    threads_parties[i].join()

        self.score = self.best_score
        return (self.best_party_id, self.best_col_id, self.best_threshold_id)

    def make_children_nodes(self, best_party_id, best_col_id, best_threshold_id):
        """
        Attach children nodes to this node.

        :param best_party_id: int
        :param best_col_id: int
        :param best_threshold_id: int
        """
        # TODO: remove idx with nan values from right_idxs
        left_idxs = self.parties[best_party_id].split_rows(
            self.idxs, best_col_id, best_threshold_id
        )
        right_idxs = [idx for idx in self.idxs if idx not in left_idxs]

        left_y_class_cnt_within_node = [0 for _ in range(self.num_classes)]
        for i in left_idxs:
            left_y_class_cnt_within_node[int(self.y[i])] += 1

        right_y_class_cnt_within_node = [0 for _ in range(self.num_classes)]
        for i in right_idxs:
            right_y_class_cnt_within_node[int(self.y[i])] += 1

        left_is_satisfied_secure_cond = is_satisfied_mi_bound(
            self.num_classes,
            self.mi_bound,
            len(left_idxs),
            len(self.y),
            self.entire_class_cnt,
            self.prior,
            left_y_class_cnt_within_node,
        )
        right_is_satisfied_secure_cond = is_satisfied_mi_bound(
            self.num_classes,
            self.mi_bound,
            len(right_idxs),
            len(self.y),
            self.entire_class_cnt,
            self.prior,
            right_y_class_cnt_within_node,
        )

        self.left = RandomForestNode(
            self.parties,
            self.y,
            self.y_onehot_encoded,
            self.num_classes,
            left_idxs,
            self.depth - 1,
            self.prior,
            self.mi_bound,
            self.active_party_id,
            (self.use_only_active_party or left_is_satisfied_secure_cond),
            self.n_job,
            self.y_onehot_encoded_encrypted,
        )
        if self.left.is_leaf_flag == 1:
            self.left.party_id = self.party_id
        self.right = RandomForestNode(
            self.parties,
            self.y,
            self.y_onehot_encoded,
            self.num_classes,
            right_idxs,
            self.depth - 1,
            self.prior,
            self.mi_bound,
            self.active_party_id,
            (self.use_only_active_party or right_is_satisfied_secure_cond),
            self.n_job,
            self.y_onehot_encoded_encrypted,
        )
        if self.right.is_leaf_flag == 1:
            self.right.party_id = self.party_id

        # Notice: this flag only supports for the case of two parties
        if (
                self.left.is_leaf_flag == 1
                and self.right.is_leaf_flag == 1
                and self.party_id == self.active_party_id
        ):
            self.left.not_splitted_flag = True
            self.right.not_splitted_flag = True

        # Clear unused index
        """
        if not (
            (self.left.not_splitted_flag and self.right.not_splitted_flag)
            or (
                self.left.secure_flag_exclude_passive_parties
                and self.right.secure_flag_exclude_passive_parties
            )
        ):
            self.idxs.clear()
            self.idxs = []
        """

    def is_leaf(self):
        if self.is_leaf_flag == -1:
            return self.is_pure() or math.isinf(self.score) or self.depth <= 0
        else:
            return self.is_leaf_flag

    def is_pure(self):
        if self.is_pure_flag == -1:
            s = set()
            for i in range(self.row_count):
                if self.y[self.idxs[i]] not in s:
                    s.add(self.y[self.idxs[i]])
                    if len(s) == 2:
                        self.is_pure_flag = 0
                        return False
            self.is_pure_flag = 1
            return True
        else:
            return self.is_pure_flag == 1

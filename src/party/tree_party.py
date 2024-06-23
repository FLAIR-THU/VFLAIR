from typing import List, Tuple

import numpy as np

from .tree_core_party import Party


class RandomForestParty(Party):
    def __init__(
            self,
            x: List[List[float]],
            num_classes: int,
            feature_id: List[int],
            party_id: int,
            min_leaf: int,
            subsample_cols: float,
            seed: int = 0,
    ):
        super().__init__(
            x, num_classes, feature_id, party_id, min_leaf, subsample_cols, False, True, seed
        )

    def get_threshold_candidates(self, x_col: List[float]) -> List[float]:
        x_col_wo_duplicates = list(set(x_col))
        threshold_candidates = x_col_wo_duplicates.copy()
        threshold_candidates.sort()
        return threshold_candidates

    def greedy_search_split(
            self, idxs: List[int], y: List[List[int]]
    ) -> List[List[Tuple[float, List[float]]]]:
        num_thresholds = self.subsample_col_count
        split_candidates_leftsize_leftposcnt = [[] for _ in range(num_thresholds)]
        self.temp_thresholds = [[] for _ in range(num_thresholds)]

        row_count = len(idxs)
        temp_y_class_cnt = [0 for _ in range(self.num_classes)]
        for r in range(row_count):
            for c in range(self.num_classes):
                temp_y_class_cnt[c] += y[r][c]

        for i in range(self.subsample_col_count):
            k = self.temp_column_subsample[i]
            x_col = np.zeros(row_count)
            not_missing_values_count = 0
            missing_values_count = 0
            for r in range(row_count):
                if not np.isnan(self.x[idxs[r], k]):
                    x_col[not_missing_values_count] = self.x[idxs[r], k]
                    not_missing_values_count += 1
                else:
                    missing_values_count += 1
            x_col = x_col[:not_missing_values_count]
            x_col_idxs = np.argsort(x_col)

            x_col = x_col[x_col_idxs]
            x_col_idxs = x_col_idxs.tolist()

            threshold_candidates = self.get_threshold_candidates(x_col)

            current_min_idx = 0
            cumulative_left_size = 0
            num_threshold_candidates = len(threshold_candidates)
            for p in range(num_threshold_candidates):
                temp_left_size = 0
                temp_left_y_class_cnt = [0 for _ in range(self.num_classes)]
                for r in range(current_min_idx, not_missing_values_count):
                    if x_col[r] <= threshold_candidates[p]:
                        for c in range(self.num_classes):
                            temp_left_y_class_cnt[c] += y[idxs[x_col_idxs[r]]][c]
                        temp_left_size += 1.0
                        cumulative_left_size += 1
                    else:
                        current_min_idx = r
                        break

                if (
                        cumulative_left_size >= self.min_leaf
                        and row_count - cumulative_left_size >= self.min_leaf
                ):
                    split_candidates_leftsize_leftposcnt[i].append(
                        (temp_left_size, temp_left_y_class_cnt)
                    )
                    self.temp_thresholds[i].append(threshold_candidates[p])

        return split_candidates_leftsize_leftposcnt


class XGBoostParty(Party):
    def __init__(
        self,
        x,
        num_classes,
        feature_id,
        party_id,
        min_leaf,
        subsample_cols,
        num_precentile_bin,
        use_missing_value=False,
        use_encrypted_label=True,
        seed=0,
    ):
        super().__init__(
            x,
            num_classes,
            feature_id,
            party_id,
            min_leaf,
            subsample_cols,
            use_missing_value,
            use_encrypted_label,
            seed,
        )
        self.num_precentile_bin = num_precentile_bin
        self.cum_num_addition = 0
        self.cum_num_communicated_ciphertexts = 0
        self.num_gss_called = 0

    def get_threshold_candidates(self, x_col):
        if len(x_col) > self.num_precentile_bin:
            return super().get_threshold_candidates(
                np.quantile(
                    x_col,
                    [
                        i / self.num_precentile_bin
                        for i in range(1, self.num_precentile_bin + 1)
                    ],
                )
            )
        else:
            return super().get_threshold_candidates(x_col)

    def greedy_search_split(self, gradient, hessian, y, idxs):
        if self.use_missing_value:
            num_thresholds = self.subsample_col_count * 2
        else:
            num_thresholds = self.subsample_col_count

        split_candidates_grad_hess = [[] for _ in range(num_thresholds)]
        self.temp_thresholds = [[] for _ in range(num_thresholds)]
        self.num_gss_called += 1

        row_count = len(idxs)
        grad_dim = len(gradient[0])
        for i in range(self.subsample_col_count):
            k = self.temp_column_subsample[i]
            x_col = []

            not_missing_values_count = 0
            missing_values_count = 0
            for r in range(row_count):
                if self.x[idxs[r]][k] is not None:
                    x_col.append(self.x[idxs[r]][k])
                    not_missing_values_count += 1
                else:
                    missing_values_count += 1

            x_col_idxs = np.argsort(x_col)
            x_col.sort()

            percentiles = self.get_threshold_candidates(x_col)

            current_min_idx = 0
            cumulative_left_size = 0
            for p in range(len(percentiles)):
                temp_grad = [0 for _ in range(grad_dim)]
                temp_hess = [0 for _ in range(grad_dim)]
                temp_left_size = 0
                temp_left_y_class_cnt = [0 for _ in range(self.num_classes)]

                for r in range(current_min_idx, not_missing_values_count):
                    if x_col[r] <= percentiles[p]:
                        if self.use_encrypted_label:
                            for c in range(self.num_classes):
                                temp_left_y_class_cnt[c] += y[idxs[x_col_idxs[r]]][c]
                        for c in range(grad_dim):
                            temp_grad[c] += gradient[idxs[x_col_idxs[r]]][c]
                            temp_hess[c] += hessian[idxs[x_col_idxs[r]]][c]
                            self.cum_num_addition += 2
                        temp_left_size += 1
                        cumulative_left_size += 1
                    else:
                        current_min_idx = r
                        break
                if (
                        cumulative_left_size >= self.min_leaf
                        and row_count - cumulative_left_size >= self.min_leaf
                ):
                    split_candidates_grad_hess[i].append(
                        (temp_grad, temp_hess, temp_left_size, temp_left_y_class_cnt)
                    )
                    self.temp_thresholds[i].append(percentiles[p])
                    self.cum_num_communicated_ciphertexts += 2 * self.num_classes

            if self.use_missing_value:
                current_max_idx = not_missing_values_count - 1
                cumulative_right_size = 0

                for p in range(len(percentiles) - 1, 0, -1):
                    temp_grad = [0 for _ in range(grad_dim)]
                    temp_hess = [0 for _ in range(grad_dim)]
                    temp_left_size = 0
                    temp_left_y_class_cnt = [0 for _ in range(self.num_classes)]

                    for r in range(current_max_idx, 0, -1):
                        if x_col[r] <= percentiles[p]:
                            if self.use_encrypted_label:
                                for c in range(self.num_classes):
                                    temp_left_y_class_cnt[c] += y[idxs[x_col_idxs[r]]][c]
                            for c in range(grad_dim):
                                temp_grad[c] += gradient[idxs[x_col_idxs[r]]][c]
                                temp_hess[c] += hessian[idxs[x_col_idxs[r]]][c]
                                self.cum_num_addition += 2
                            temp_left_size += 1
                            cumulative_right_size += 1
                        else:
                            current_max_idx = r
                            break
                    if (
                            cumulative_right_size >= self.min_leaf
                            and row_count - cumulative_right_size >= self.min_leaf
                    ):
                        split_candidates_grad_hess[i + self.subsample_col_count].append(
                            (
                                temp_grad,
                                temp_hess,
                                temp_left_size,
                                temp_left_y_class_cnt
                            )
                        )
                        self.temp_thresholds[i + self.subsample_col_count].append(
                            percentiles[p]
                        )

        return split_candidates_grad_hess

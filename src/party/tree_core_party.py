from joblib import Parallel, delayed
import random

import numpy as np


class Party:
    def __init__(
        self,
        x,
        num_classes,
        feature_id,
        party_id,
        min_leaf,
        subsample_cols,
        use_missing_value=False,
        use_encrypted_label=True,
        seed=0,
    ):
        self.x = x
        self.num_classes = num_classes
        self.feature_id = feature_id
        self.party_id = party_id
        self.min_leaf = min_leaf
        self.subsample_cols = subsample_cols
        self.use_missing_value = use_missing_value
        self.use_encrypted_label = use_encrypted_label
        self.seed = seed

        self.col_count = len(x[0])
        self.subsample_col_count = max(1, int(subsample_cols * self.col_count))

        self.lookup_table = {}
        self.temp_column_subsample = []
        self.temp_thresholds = []

        self.pk = None
        self.sk = None

        random.seed(self.seed)

    def set_keypair(self, pk, sk):
        self.pk = pk
        self.sk = sk

    def encrypt_row(self, row):
        return [self.pk.encrypt(e) for e in row]

    def encrypt_2dlist(self, x):
        results = Parallel(n_jobs=-1)(delayed(self.encrypt_row)(row) for row in x)
        return results

    def decrypt_1dlist(self, row):
        results = []
        for e in row:
            if e == 0:
                results.append(e)
            else:
                try:
                    results.append(self.sk.decrypt(e))
                except Exception as e:
                    results.append(0)
                    print(type(e), str(e))
                    continue
        return results

    def decrypt_2dlist(self, mat):
        results = Parallel(n_jobs=-1)(delayed(self.decrypt_1dlist)(row) for row in mat)
        return results

    def get_lookup_table(self):
        return self.lookup_table

    def get_threshold_candidates(self, x_col):
        x_col_wo_duplicates = np.unique(x_col)
        return x_col_wo_duplicates

    def is_left(self, record_id, xi):
        x_criterion = xi[self.feature_id[self.lookup_table[record_id][0]]]
        if x_criterion is not None:
            flag = x_criterion <= self.lookup_table[record_id][1]
        else:
            flag = self.lookup_table[record_id][2] == 0
        return flag

    def subsample_columns(self):
        self.temp_column_subsample = list(range(self.col_count))
        random.shuffle(self.temp_column_subsample)

    def split_rows(self, idxs, feature_opt_pos, threshold_opt_pos):
        feature_opt_id = self.temp_column_subsample[
            feature_opt_pos % self.subsample_col_count
            ]
        if feature_opt_pos > self.subsample_col_count:
            missing_dir = 1
        else:
            missing_dir = 0
        row_count = len(idxs)
        x_col = [self.x[idxs[r]][feature_opt_id] for r in range(row_count)]
        threshold = self.temp_thresholds[feature_opt_pos][threshold_opt_pos]
        left_idxs = [
            idxs[r]
            for r in range(row_count)
            if ((x_col[r] is not None) and (x_col[r] <= threshold))
               or ((x_col[r] is None) and (missing_dir == 1))
        ]
        return left_idxs

    def insert_lookup_table(self, feature_opt_pos, threshold_opt_pos):
        feature_opt_id = self.temp_column_subsample[
            feature_opt_pos % self.subsample_col_count
            ]
        threshold_opt = self.temp_thresholds[feature_opt_pos][threshold_opt_pos]

        if self.use_missing_value:
            if feature_opt_pos > self.subsample_col_count:
                missing_dir = 1
            else:
                missing_dir = 0

        else:
            missing_dir = -1

        self.lookup_table[len(self.lookup_table)] = (
            feature_opt_id,
            threshold_opt,
            missing_dir,
        )
        return len(self.lookup_table) - 1

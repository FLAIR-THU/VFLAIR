import os
import sys

sys.path.append(os.pardir)

import numpy as np
from tree_party import Party


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
            seed,
        )
        self.num_precentile_bin = num_precentile_bin

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

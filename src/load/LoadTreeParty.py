from party.tree_party import RandomForestParty, XGBoostParty
import numpy as np
import os
import sys

sys.path.append(os.pardir)


def load_tree_parties(args):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k
    assert args.k > 1

    num_classes = len(np.unique(args.y))

    if args.model_type == "xgboost":
        for ik in range(args.k):
            args.parties[ik] = XGBoostParty(
                args.datasets[ik],
                num_classes,
                args.featureid_lists[ik],
                ik,
                args.min_leaf,
                args.subsample_cols,
                args.max_bin,
                args.use_missing_value,
                args.use_encrypted_label,
                args.seed,
            )
    elif args.model_type == "randomforest":
        for ik in range(args.k):
            args.parties[ik] = RandomForestParty(
                args.datasets[ik],
                num_classes,
                args.featureid_lists[ik],
                ik,
                args.min_leaf,
                args.subsample_cols,
                args.seed,
            )
    else:
        raise ValueError(f"model_type should be `xgboost` or `randomforest`")

    return args

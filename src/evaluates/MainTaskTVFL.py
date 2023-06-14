import os
import sys

sys.path.append(os.pardir)

import random

from phe import paillier

from models.tree import *
from party.tree_party import *


class MainTaskTVFL(object):
    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.parties = args.parties
        self.y = args.y
        self.num_classes = len(np.unique(self.y))
        self.model_type = args.model_type

        self.use_encryption = args.use_encryption
        self.key_length = args.key_length

        self.seed_base = args.seed_base
        self.number_of_trees = args.number_of_trees
        self.depth = args.depth

    def setup_keypair(self):
        public_key, private_key = paillier.generate_paillier_keypair(
            n_length=self.key_length
        )
        self.parties[self.k - 1].set_keypair(public_key, private_key)

    def train(self):
        if self.use_encryption:
            self.setup_keypair()

        random.seed(self.seed_base)

        if self.model_type == "xgboost":
            self.clf = XGBoostClassifier(
                self.num_classes,
                boosting_rounds=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.k - 1,
                use_encryption=self.use_encryption,
            )
        elif self.model_type == "randomforest":
            self.clf = RandomForestClassifier(
                self.num_classes,
                num_trees=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.k - 1,
                use_encryption=self.use_encryption,
            )
        else:
            raise ValueError(f"model_type should be `xgboost` or `randomforest`")

        self.clf.fit(self.parties, self.y)

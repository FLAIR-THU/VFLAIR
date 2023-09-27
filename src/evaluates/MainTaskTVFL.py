from party.tree_party import *
from models.tree import *
from phe import paillier
import random
import os
import sys

sys.path.append(os.pardir)


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

        self.seed = args.seed
        self.number_of_trees = args.number_of_trees
        self.depth = args.depth

        self.advanced_params = args.advanced_params

    def setup_keypair(self):
        public_key, private_key = paillier.generate_paillier_keypair(
            n_length=self.key_length
        )
        self.parties[self.k - 1].set_keypair(public_key, private_key)

    def train(self):
        if self.use_encryption:
            self.setup_keypair()

        if self.model_type == "xgboost":
            self.clf = XGBoostClassifier(
                self.num_classes,
                boosting_rounds=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.k - 1,
                use_encryption=self.use_encryption,
                **self.advanced_params
            )
        elif self.model_type == "randomforest":
            self.clf = RandomForestClassifier(
                self.num_classes,
                num_trees=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.k - 1,
                use_encryption=self.use_encryption,
                **self.advanced_params
            )
        else:
            raise ValueError(f"model_type should be `xgboost` or `randomforest`")

        random.seed(self.seed)

        self.clf.fit(self.parties, self.y)

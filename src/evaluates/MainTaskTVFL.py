from party.tree_party import *
from evaluates.defenses.lpmst import *
from evaluates.defenses.grafting import *
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

        self.apply_defense = args.apply_defense
        self.defense_name = args.defense_name
        self.lpmst_eps = args.lpmst_eps
        self.lpmst_m = args.lpmst_m

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

        random.seed(self.seed)

        if self.apply_defense:
            if self.defense_name == "grafting-ldp":
                lpmst = LPMST(self.lpmst_m, self.lpmst_eps, 0)
                lpmst.fit(self.clf, self.parties, self.y)
                grafting_forest(self.clf, self.y)
            elif self.defense_name == "idlmid":
                pass
            else:
                raise ValueError(f"defense_name should be `grafting-ldp` or `idlmid`")
        else:
            self.clf.fit(self.parties, self.y)

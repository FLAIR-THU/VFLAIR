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
        self.active_party_id = args.active_party_id
        self.parties = args.parties
        self.y = args.y
        self.num_classes = len(np.unique(self.y))
        self.model_type = args.model_type

        self.use_encryption = args.use_encryption
        self.key_length = args.key_length

        self.seed = args.seed
        self.number_of_trees = args.number_of_trees
        self.depth = args.depth
        self.is_hybrid = args.is_hybrid

        self.apply_defense = args.apply_defense
        self.defense_name = args.defense_name
        self.lpmst_eps = args.lpmst_eps
        self.lpmst_m = args.lpmst_m
        self.mi_bound = args.mi_bound
        self.advanced_params = args.advanced_params

    def setup_keypair(self):
        public_key, private_key = paillier.generate_paillier_keypair(
            n_length=self.key_length
        )
        self.parties[self.active_party_id].set_keypair(public_key, private_key)

    def train(self):
        if self.use_encryption:
            self.setup_keypair()

        if self.model_type == "xgboost":
            self.clf = XGBoostClassifier(
                self.num_classes,
                boosting_rounds=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.active_party_id,
                use_encryption=self.use_encryption,
                is_hybrid=self.is_hybrid,
                **self.advanced_params
            )
        elif self.model_type == "randomforest":
            self.clf = RandomForestClassifier(
                self.num_classes,
                num_trees=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.active_party_id,
                use_encryption=self.use_encryption,
                **self.advanced_params
            )
        else:
            raise ValueError(f"model_type should be `xgboost` or `randomforest`")

        random.seed(self.seed)

        if self.apply_defense:
            if self.defense_name == "grafting-ldp":
                lpmst = LPMST(self.lpmst_m, self.lpmst_eps, 0)
                lpmst.fit(self.clf, self.parties, self.y)
                grafting_forest(self.clf, self.y)
            elif self.defense_name == "lp-mst":
                lpmst = LPMST(self.lpmst_m, self.lpmst_eps, 0)
                lpmst.fit(self.clf, self.parties, self.y)
            elif self.defense_name == "id-lmid":
                self.clf.mi_bound = self.mi_bound
                self.clf.fit(self.parties, self.y)
            else:
                raise ValueError(f"defense_name should be `grafting-ldp`, `id-lmid`, or `lp-mst`")
        else:
            self.clf.fit(self.parties, self.y)

        if self.model_type == "xgboost":
            for i, p in enumerate(self.parties):
                print(f" party-{i}: cum_num_addition={p.cum_num_addition}")
                print(f" party-{i}: num_gss_called={p.num_gss_called}")
                print(f" party-{i}: cum_num_communicated_ciphertexts={p.cum_num_communicated_ciphertexts}")
            print(f" time spent for encryption: {self.clf.cum_time_encryption}")

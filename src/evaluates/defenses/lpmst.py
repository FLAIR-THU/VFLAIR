import random
import math

import numpy as np


class RRWithPrior:
    def __init__(self, epsilon, prior_dist, seed=0):
        self.epsilon = epsilon
        self.K = len(prior_dist)
        self.prior_dist = prior_dist

        self._set_random(seed)
        self._set_label2argmaxpos()
        self.search_optimal_k()

    def _set_random(self, seed):
        random.seed(seed)

    def _set_label2argmaxpos(self):
        self.sort_idx = list(range(self.K))
        self.sort_idx.sort(key=lambda i: -self.prior_dist[i])
        self.label2argmaxpos = {label: k for k, label in enumerate(self.sort_idx)}

    def search_optimal_k(self):
        max_w_k = 0
        cumulative_p = 0
        exp_eps = math.exp(self.epsilon)
        for k in range(self.K):
            cumulative_p += self.prior_dist[self.sort_idx[k]]
            temp_w_k = exp_eps / (exp_eps + k) * cumulative_p

            if temp_w_k > max_w_k:
                max_w_k = temp_w_k
                self.k_star = k + 1

        self.best_w_k = max_w_k
        self.threshold_prob = exp_eps / (exp_eps + self.k_star - 1)

    def rrtop_k(self, y):
        y_random = 0
        temp_idx = 0
        if self.label2argmaxpos[y] < self.k_star:
            p = random.uniform(0, 1)
            if self.threshold_prob > p:
                y_random = y
            else:
                temp_idx = random.randint(1, self.k_star - 1)
                y_random = self.sort_idx[temp_idx - 1]
                if y_random == y:
                    y_random = self.sort_idx[self.k_star - 1]
        else:
            y_random = random.randint(0, self.K - 1)
        return y_random


class LPMST:
    def __init__(self, M=1, epsilon=1.0, seed=0):
        self.M = M
        self.epsilon = epsilon
        self.seed = seed
        self.rrp = None

    def fit(self, clf, parties, y):
        n = len(y)
        chunk_size = n // self.M
        class_num = max(y) + 1
        init_prior_dist = [1.0 / class_num for _ in range(class_num)]

        temp_ptr = 0
        temp_seed = self.seed

        for m in range(self.M):
            if m == 0:
                self.rrp = RRWithPrior(self.epsilon, init_prior_dist, self.seed)
                y_hat = [self.rrp.rrtop_k(y[i]) for i in range(temp_ptr, chunk_size)]
                temp_ptr = chunk_size
            else:
                temp_party_vec = [parties[clf.active_party_id]]
                clf.clear()
                clf.fit(temp_party_vec, y_hat)
                for i in range(temp_ptr, min(n, chunk_size * (m + 1))):
                    temp_x = parties[clf.active_party_id].x[i].reshape(1, -1)
                    prior_dist = clf.predict_proba(temp_x)[0]
                    self.rrp = RRWithPrior(self.epsilon, prior_dist, temp_seed)
                    temp_seed += 1
                    y_hat.append(self.rrp.rrtop_k(y[i]))
                temp_ptr = min(n, chunk_size * (m + 1))

        clf.clear()
        clf.fit(parties, y_hat)

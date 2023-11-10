# README for Tree-based Vertical Federated Learning

We provide support for Tree-based Vertical Federated Learning (T-VFL), implementing a T-VFL framework similar to XGBoost, as described in [Kewei et al (2019)](https://arxiv.org/pdf/1901.08755.pdf). We have also made a few modifications to it and additionally support RandomForest.

## How to write a configuration file

### 1. Basic Parameters

- "tree\_type": type of a model. `xgboost` or `randomforest`
- "number\_of\_trees": number of trees (boosting rounds).
- "depth": maximum depth.
- "min\_leaf": minimum number of samples on a leaf.
- "subsample\_cols": subsample ratio of the features.
- "max\_bin": maximum number of discrete bins to group features into buckets.
- "use\_missing\_value": if true, consider missing values (only for xgboost).
- "k": number of trees.
- "use\_encryption": if true, use paillier encryption.
- "key\_length": key length of paillier encryption.

### 2. Defense

- LP-MST ([Deep Learning with Label Differential Privacy](https://arxiv.org/pdf/2102.06062.pdf))

```json
 "defense": {
    "name": "lp-mst",
    "parameters": {"lpmst_eps": 0.3}
 }
```

- Grafting-LDP ([ELIMINATING LABEL LEAKAGE IN TREE-BASED VERTICAL FEDERATED LEARNING](https://arxiv.org/pdf/2307.10318.pdf))

> Grafting-LDP is implemented on the top of LP-MST.

```json
 "defense": {
    "name": "grafting-ldp",
    "parameters": {"lpmst_eps": 0.3}
 }
```

- ID-LMID ([ELIMINATING LABEL LEAKAGE IN TREE-BASED VERTICAL FEDERATED LEARNING](https://arxiv.org/pdf/2307.10318.pdf))

```json
 "defense": {
    "name": "id-lmid",
    "parameters": {"mi_bound": 0.1}
 }
```


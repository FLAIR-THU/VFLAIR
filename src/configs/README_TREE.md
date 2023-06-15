# README for Tree-based Vertical Federated Learning

We provide support for Tree-based Vertical Federated Learning (T-VFL), implementing a T-VFL framework similar to XGBoost, as described in [Kewei et al (2019)](https://arxiv.org/pdf/1901.08755.pdf). We have also made a few modifications to it and additionally support RandomForest.

## One. Basic Parameters

- "model\_type": type of a model. `xgboost` or `randomforest`
- "number\_of\_trees": number of trees (boosting rounds).
- "depth": maximum depth.
- "min\_leaf": minimum number of samples on a leaf.
- "subsample\_cols": subsample ratio of the features.
- "max\_bin": maximum number of discrete bins to group features into buckets.
- "use\_missing\_value": if true, consider missing values (only for xgboost).
- "k": number of trees.
- "use\_encryption": if true, use paillier encryption.
- "key\_length": key length of paillier encryption.

# ID2Graph against Tree-based algorithm

This attack steals training labels from the trained tree-based model via the instance space. This attack can apply to many tree-based VFL methods revealing the instance spaces. We support both RandomForest and XGBoost.

## Example

You can run the example experiment with `main_id2graph.py`.

```
cd src
python main_id2graph.py
```

### Usage

```
usage: id2graph [-h] [--model MODEL] [--num_trials NUM_TRIALS] [--eta ETA] [--min_leaf MIN_LEAF]
                [--subsample_cols SUBSAMPLE_COLS] [--max_bin MAX_BIN] [--depth DEPTH]
                [--number_of_trees NUMBER_OF_TREES] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         attack xgboost or randomforest
  --num_trials NUM_TRIALS
                        number of trials
  --eta ETA             discount factor
  --min_leaf MIN_LEAF   minmum number of samples within a leaf
  --subsample_cols SUBSAMPLE_COLS
                        subsampling ratio of features used for training each tree
  --max_bin MAX_BIN     maximum number of bins (used only for xgboost)
  --depth DEPTH         maximum depth
  --number_of_trees NUMBER_OF_TREES
                        number of trees
  --seed SEED           random seed
```
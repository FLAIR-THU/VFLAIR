#echo "Nursery - RandomForest"
#for seed in 1 2 3 4 5
#do
#    python3 main_tree.py --dataset nursery --configs tree/basic_configs_randomforest --seed ${seed}
#done

#echo "Nursery - XGBoost"
#for seed in 1 2 3 4 5
#do
#    python3 main_tree.py --dataset nursery --configs tree/basic_configs_xgboost --seed ${seed}
#done

#echo "Credit - RandomForest"
#for seed in 1 2 3 4 5
#do
#    python3 main_tree.py --dataset credit --configs tree/basic_configs_randomforest --seed ${seed}
#done

echo "Credit - XGBoost"
for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset credit --configs tree/basic_configs_xgboost --seed ${seed} --grid
done

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset nursery --configs tree/basic_configs_randomforest --seed ${seed} >> tmpoutput/output_nr.out &
done

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset nursery --configs tree/basic_configs_secureforest --seed ${seed} >> tmpoutput/output_nsr.out &
done

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset nursery --configs tree/basic_configs_xgboost --seed ${seed} >> tmpoutput/output_nx.out &
done

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset nursery --configs tree/basic_configs_secureboost --seed ${seed} >> tmpoutput/output_nsx.out &
done
echo "Nursery" & wait

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset credit --configs tree/basic_configs_randomforest --seed ${seed} --grid >> tmpoutput/output_cr.out &
done

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset credit --configs tree/basic_configs_secureforest --seed ${seed} --grid >> tmpoutput/output_csr.out &
done

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset credit --configs tree/basic_configs_xgboost --seed ${seed} --grid >> tmpoutput/output_cx.out &
done

for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset credit --configs tree/basic_configs_secureboost --seed ${seed} --grid >> tmpoutput/output_csx.out &
done
echo "Credit" & wait

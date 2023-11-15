for seed in 1 2 3 4 5
do
    python3 main_tree.py --dataset credit --configs tree/basic_configs_secureforest --seed ${seed} --grid >> tmpoutput/output_csr.out &
done

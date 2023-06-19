#!/bin/bash
# README: run this file under src folder

# ###### make sure starts with non-trainable head ######
sed -i 's/"lr": 0.050/"lr": 0.001/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.9/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 99.9/"gradient_sparse_rate": 99.5/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 100.0/g' ./configs/test/mnist_bli_gs_configs.json


# ###### change to trainable head ######
sed -i 's/"lr": 0.001/"lr": 0.050/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"apply_trainable_layer": 0/"apply_trainable_layer": 1/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"global_model": "ClassificationModelHostHead"/"global_model": "ClassificationModelHostTrainableHead"/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.9/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 99.9/"gradient_sparse_rate": 99.5/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/test/mnist_bli_gs_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_gs_configs --seed $i
done

sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 100.0/g' ./configs/test/mnist_bli_gs_configs.json

sed -i 's/"lr": 0.050/"lr": 0.001/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_bli_gs_configs.json

# srun --gres=gpu:a100-80G:1 --time=1-00:00:00 --mem=20000 bash ./local_test/mnist_bli_gs.sh
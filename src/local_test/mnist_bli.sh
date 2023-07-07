#!/bin/bash
# README: run this file under src folder

# ###### make sure starts with non-trainable head ######
sed -i 's/"lr": 0.050/"lr": 0.001/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_bli_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_bli_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_configs --seed $i
done

# ###### change to trainable head ######
sed -i 's/"lr": 0.001/"lr": 0.05/g' ./configs/test/mnist_bli_configs.json
sed -i 's/"apply_trainable_layer": 0/"apply_trainable_layer": 1/g' ./configs/test/mnist_bli_configs.json
sed -i 's/"global_model": "ClassificationModelHostHead"/"global_model": "ClassificationModelHostTrainableHead"/g' ./configs/test/mnist_bli_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_bli_configs --seed $i
done

# ###### change back to non-trainable head ######
sed -i 's/"lr": 0.050/"lr": 0.001/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_bli_gs_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_bli_gs_configs.json

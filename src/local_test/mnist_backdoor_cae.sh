#!/bin/bash
# README: run this file under src folder
# for cae+Backdoor, mai_lr fix to 0.001; w/o top_model, lambda change from [1,0] with model path specified manually;


# ###### make sure starts with non-trainable head ######
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_1.0_1642396548"|"model_path": "../trained_CAE_models/autoencoder_10_0.5_1642396797"|g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.5_1642396797"|"model_path": "../trained_CAE_models/autoencoder_10_0.1_1642396928"|g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.1_1642396928"|"model_path": "../trained_CAE_models/autoencoder_10_0.0_1631093149"|g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.0_1631093149"|"model_path": "../trained_CAE_models/autoencoder_10_1.0_1642396548"|g' ./configs/test/mnist_backdoor_cae_configs.json


# ###### change to trainable head ######
sed -i 's/"apply_trainable_layer": 0/"apply_trainable_layer": 1/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's/"global_model": "ClassificationModelHostHead"/"global_model": "ClassificationModelHostTrainableHead"/g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_1.0_1642396548"|"model_path": "../trained_CAE_models/autoencoder_10_0.5_1642396797"|g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.5_1642396797"|"model_path": "../trained_CAE_models/autoencoder_10_0.1_1642396928"|g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.1_1642396928"|"model_path": "../trained_CAE_models/autoencoder_10_0.0_1631093149"|g' ./configs/test/mnist_backdoor_cae_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_pipeline.py --configs test/mnist_backdoor_cae_configs --seed $i
done
sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.0_1631093149"|"model_path": "../trained_CAE_models/autoencoder_10_1.0_1642396548"|g' ./configs/test/mnist_backdoor_cae_configs.json


sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_backdoor_cae_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_backdoor_cae_configs.json


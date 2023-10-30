#!/bin/bash
#SBATCH --job-name dcae0_amc              # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo  'DCAE cifar100 amc Begin'
#CAE 1.0
python cifar100_amc.py --configs cifar100_amc/dcae

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/cifar100_amc/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_1.0_1698171384"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_0.5_1698170879"|g' ./configs/cifar100_amc/dcae.json
python cifar100_amc.py --configs cifar100_amc/dcae

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/cifar100_amc/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_0.5_1698170879"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_0.1_1698171291"|g' ./configs/cifar100_amc/dcae.json
python cifar100_amc.py --configs cifar100_amc/dcae

# CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/cifar100_amc/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_0.1_1698171291"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_0.0_1698171184"|g' ./configs/cifar100_amc/dcae.json
python cifar100_amc.py --configs cifar100_amc/dcae


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/cifar100_amc/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_0.0_1698171184"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_100_1.0_1698171384"|g' ./configs/cifar100_amc/dcae.json

echo  'DCAE End'
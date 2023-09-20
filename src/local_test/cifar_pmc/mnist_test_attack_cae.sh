#!/bin/bash
#SBATCH --job-name cae0_pmc             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo  'CAE cifar pmc Begin' #SBATCH --qos high
#CAE 1.0
python cifar_pmc.py --configs cifar_pmc/cae

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/cifar_pmc/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_1.0_1642396548"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_0.5_1642396797"|g' ./configs/cifar_pmc/cae.json
python cifar_pmc.py --configs cifar_pmc/cae

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/cifar_pmc/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_0.5_1642396797"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_0.1_1686717045"|g' ./configs/cifar_pmc/cae.json
python cifar_pmc.py --configs cifar_pmc/cae

# CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/cifar_pmc/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_0.1_1686717045"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_0.0_1631093149"|g' ./configs/cifar_pmc/cae.json
python cifar_pmc.py --configs cifar_pmc/cae


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/cifar_pmc/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_0.0_1631093149"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_10_1.0_1642396548"|g' ./configs/cifar_pmc/cae.json

echo  'CAE End'
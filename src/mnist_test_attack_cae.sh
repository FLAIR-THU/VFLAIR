#!/bin/bash
#SBATCH --job-name cae0_main             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# echo  'CAE Begin' #SBATCH --qos high
# #CAE 1.0
# python main_pipeline.py --configs main_split/mnist_test_attack_cae --gpu 1

# # # CAE 0.5
# sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/main_split/mnist_test_attack_cae.json
# sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_1.0_1642396548"|"model_path": "../trained_CAE_models/autoencoder_10_0.5_1642396797"|g' ./configs/main_split/mnist_test_attack_cae.json
# python main_pipeline.py --configs main_split/mnist_test_attack_cae --gpu 1

# # # CAE 0.1
# sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/main_split/mnist_test_attack_cae.json
# sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.5_1642396797"|"model_path": "../trained_CAE_models/autoencoder_10_0.1_1686717045"|g' ./configs/main_split/mnist_test_attack_cae.json
# python main_pipeline.py --configs main_split/mnist_test_attack_cae --gpu 1

# # CAE 0.0
# sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/main_split/mnist_test_attack_cae.json
# sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.1_1686717045"|"model_path": "../trained_CAE_models/autoencoder_10_0.0_1631093149"|g' ./configs/main_split/mnist_test_attack_cae.json
# python main_pipeline.py --configs main_split/mnist_test_attack_cae --gpu 1


# sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/main_split/mnist_test_attack_cae.json
# sed -i 's|"model_path": "../trained_CAE_models/autoencoder_10_0.0_1631093149"|"model_path": "../trained_CAE_models/autoencoder_10_1.0_1642396548"|g' ./configs/main_split/mnist_test_attack_cae.json

# echo  'CAE End'

#CAE 1.0
python main_pipeline.py --configs main/mnist_test_attack_cae --gpu 1

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/main/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_1.0_1630879452"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_0.5_1630891447"|g' ./configs/main/mnist_test_attack_cae.json
python main_pipeline.py --configs main/mnist_test_attack_cae --gpu 1

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/main/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_0.5_1630891447"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_0.1_1630895642"|g' ./configs/main/mnist_test_attack_cae.json
python main_pipeline.py --configs main/mnist_test_attack_cae --gpu 1

# # CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/main/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_0.1_1630895642"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_0.0_1631059899"|g' ./configs/main/mnist_test_attack_cae.json
python main_pipeline.py --configs main/mnist_test_attack_cae --gpu 1


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/main/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_0.0_1631059899"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_5_1.0_1630879452"|g' ./configs/main/mnist_test_attack_cae.json

echo  'CAE End'

echo  'DCAE main renew Begin'
#CAE 1.0
python main_pipeline.py --configs main/mnist_test_attack_dcae --gpu 1

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/main/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_1.0_1642396548"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.5_1642396797"|g' ./configs/main/mnist_test_attack_dcae.json
python main_pipeline.py --configs main/mnist_test_attack_dcae --gpu 1

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/main/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.5_1642396797"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.1_1686717045"|g' ./configs/main/mnist_test_attack_dcae.json
python main_pipeline.py --configs main/mnist_test_attack_dcae --gpu 1

# CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/main/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.1_1686717045"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.0_1631093149"|g' ./configs/main/mnist_test_attack_dcae.json
python main_pipeline.py --configs main/mnist_test_attack_dcae --gpu 1


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/main/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.0_1631093149"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_1.0_1642396548"|g' ./configs/main/mnist_test_attack_dcae.json

echo  'DCAE End'
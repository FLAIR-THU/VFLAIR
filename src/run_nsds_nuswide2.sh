#!/bin/bash
#SBATCH --job-name gp0_nsds            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'GradPerturb main Begin'

# DP 0.01
python main_pipeline.py --configs nsds/mnist_test_attack_gp --gpu 7

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/nsds/mnist_test_attack_gp.json
python main_pipeline.py --configs nsds/mnist_test_attack_gp --gpu 7

# DP 0.3
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 0.3/g' ./configs/nsds/mnist_test_attack_gp.json
python main_pipeline.py --configs nsds/mnist_test_attack_gp --gpu 7

# DP 1.0
sed -i 's/"perturb_epsilon": 0.3/"perturb_epsilon": 1.0/g' ./configs/nsds/mnist_test_attack_gp.json
echo 'GradPerturb 1.0'
python main_pipeline.py --configs nsds/mnist_test_attack_gp --gpu 7

# DP 3.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 3.0/g' ./configs/nsds/mnist_test_attack_gp.json
python main_pipeline.py --configs nsds/mnist_test_attack_gp --gpu 7

# DP 10.0
sed -i 's/"perturb_epsilon": 3.0/"perturb_epsilon": 10.0/g' ./configs/nsds/mnist_test_attack_gp.json
python main_pipeline.py --configs nsds/mnist_test_attack_gp --gpu 7

sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 0.01/g' ./configs/nsds/mnist_test_attack_gp.json

echo 'GradPerturb End'


#!/bin/bash
#SBATCH --job-name dp0_nsds             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'DP agg nsds Begin'

# DP 0.0001
python main_pipeline.py --configs nsds/mnist_test_attack_dp --gpu 7

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/nsds/mnist_test_attack_dp.json
python main_pipeline.py --configs nsds/mnist_test_attack_dp --gpu 7

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/nsds/mnist_test_attack_dp.json
python main_pipeline.py --configs nsds/mnist_test_attack_dp --gpu 7

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/nsds/mnist_test_attack_dp.json
echo 'DP 0.1'
python main_pipeline.py --configs nsds/mnist_test_attack_dp --gpu 7

sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/nsds/mnist_test_attack_dp.json

echo 'DP End'

#!/bin/bash
#SBATCH --job-name dcor0_nsds            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'dCor agg nsds Begin'
# Begin with 0.0001 #SBATCH --qos high

# dCor 0.0001
echo '0.0001'
python main_pipeline.py --configs nsds/mnist_test_attack_dcor --gpu 7

# dCor 0.001
sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/nsds/mnist_test_attack_dcor.json
python main_pipeline.py --configs nsds/mnist_test_attack_dcor --gpu 7

# dCor 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/nsds/mnist_test_attack_dcor.json
python main_pipeline.py --configs nsds/mnist_test_attack_dcor --gpu 7

# DP 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/nsds/mnist_test_attack_dcor.json
python main_pipeline.py --configs nsds/mnist_test_attack_dcor --gpu 7

# DP 0.3
sed -i 's/"lambda": 0.1/"lambda": 0.3/g' ./configs/nsds/mnist_test_attack_dcor.json
python main_pipeline.py --configs nsds/mnist_test_attack_dcor --gpu 7

sed -i 's/"lambda": 0.3/"lambda": 0.0001/g' ./configs/nsds/mnist_test_attack_dcor.json

echo 'dCor End'

#!/bin/bash
#SBATCH --job-name dcae0_nsds              # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high
echo 'DCAE agg nsds renew Begin'
# DCAE 1.0 
python main_pipeline.py --configs nsds/mnist_test_attack_dcae --gpu 7

# DCAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/nsds/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|g' ./configs/nsds/mnist_test_attack_dcae.json
echo 'dcae 0.5'
python main_pipeline.py --configs nsds/mnist_test_attack_dcae --gpu 7

# DCAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/nsds/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|g' ./configs/nsds/mnist_test_attack_dcae.json
python main_pipeline.py --configs nsds/mnist_test_attack_dcae --gpu 7

# DCAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/nsds/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|g' ./configs/nsds/mnist_test_attack_dcae.json
python main_pipeline.py --configs nsds/mnist_test_attack_dcae --gpu 7


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/nsds/mnist_test_attack_dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704"|g' ./configs/nsds/mnist_test_attack_dcae.json

echo 'DCAE End'


#!/bin/bash
#SBATCH --job-name cae0_nsds              # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo  'CAE agg nsds Begin' #SBATCH --qos high
#CAE 1.0
python main_pipeline.py --configs nsds/mnist_test_attack_cae --gpu 7

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/nsds/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|g' ./configs/nsds/mnist_test_attack_cae.json
python main_pipeline.py --configs nsds/mnist_test_attack_cae --gpu 7

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/nsds/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|g' ./configs/nsds/mnist_test_attack_cae.json
python main_pipeline.py --configs nsds/mnist_test_attack_cae --gpu 7

# CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/nsds/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|g' ./configs/nsds/mnist_test_attack_cae.json
python main_pipeline.py --configs nsds/mnist_test_attack_cae --gpu 7


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/nsds/mnist_test_attack_cae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|"model_path": ""./evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704""|g' ./configs/nsds/mnist_test_attack_cae.json

echo  'CAE End'
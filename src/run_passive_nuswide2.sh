#!/bin/bash
#SBATCH --job-name gp0_nsds            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


#!/bin/bash
#SBATCH --job-name ldp0_nsds            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high

# Begin with LaplaceDP 0.0001 
echo 'LDP agg nsds Begin'
# DP 0.0001
python main_pipeline.py --configs main_passive/mnist_test_attack_ldp --gpu 7

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/main_passive/mnist_test_attack_ldp.json
python main_pipeline.py --configs main_passive/mnist_test_attack_ldp --gpu 7

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/main_passive/mnist_test_attack_ldp.json
python main_pipeline.py --configs main_passive/mnist_test_attack_ldp --gpu 7

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/main_passive/mnist_test_attack_ldp.json
python main_pipeline.py --configs main_passive/mnist_test_attack_ldp --gpu 7


sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/main_passive/mnist_test_attack_ldp.json


#!/bin/bash
#SBATCH --job-name dp0_nsds             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'DP agg nsds Begin'

# DP 0.0001
python main_pipeline.py --configs main_passive/mnist_test_attack_dp --gpu 7

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/main_passive/mnist_test_attack_dp.json
python main_pipeline.py --configs main_passive/mnist_test_attack_dp --gpu 7

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/main_passive/mnist_test_attack_dp.json
python main_pipeline.py --configs main_passive/mnist_test_attack_dp --gpu 7

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/main_passive/mnist_test_attack_dp.json
echo 'DP 0.1'
python main_pipeline.py --configs main_passive/mnist_test_attack_dp --gpu 7

sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/main_passive/mnist_test_attack_dp.json

echo 'DP End'

#!/bin/bash
#SBATCH --job-name dcor0_nsds            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'dCor agg nsds Begin'
# Begin with 0.0001 #SBATCH --qos high

# dCor 0.0001
echo '0.0001'
python main_pipeline.py --configs main_passive/mnist_test_attack_dcor --gpu 7

# dCor 0.001
sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/main_passive/mnist_test_attack_dcor.json
python main_pipeline.py --configs main_passive/mnist_test_attack_dcor --gpu 7

# dCor 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/main_passive/mnist_test_attack_dcor.json
python main_pipeline.py --configs main_passive/mnist_test_attack_dcor --gpu 7

# DP 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/main_passive/mnist_test_attack_dcor.json
python main_pipeline.py --configs main_passive/mnist_test_attack_dcor --gpu 7

# DP 0.3
sed -i 's/"lambda": 0.1/"lambda": 0.3/g' ./configs/main_passive/mnist_test_attack_dcor.json
python main_pipeline.py --configs main_passive/mnist_test_attack_dcor --gpu 7

sed -i 's/"lambda": 0.3/"lambda": 0.0001/g' ./configs/main_passive/mnist_test_attack_dcor.json

echo 'dCor End'

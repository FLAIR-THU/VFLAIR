#!/bin/bash
#SBATCH --job-name dcor0_main            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'dCor/gp split blidli Begin '
# Begin with 0.0001 #SBATCH --qos high
# DP 0.0001
echo '0.0001'
python cifar100_blidli.py --configs cifar100_blidli/dcor
sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/cifar100_blidli/dcor.json

# DP 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/cifar100_blidli/dcor.json
python cifar100_blidli.py --configs cifar100_blidli/dcor

# DP 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/cifar100_blidli/dcor.json
python cifar100_blidli.py --configs cifar100_blidli/dcor

# 0.3
sed -i 's/"lambda": 0.1/"lambda": 0.3/g' ./configs/cifar100_blidli/dcor.json
python cifar100_blidli.py --configs cifar100_blidli/dcor

sed -i 's/"lambda": 0.3/"lambda": 0.0001/g' ./configs/cifar100_blidli/dcor.json

echo 'dCor End'

#!/bin/bash
#SBATCH --job-name gs0_amc              # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'GS90 80  dp   gp Begin'
# 100.0 #SBATCH --qos high
# python cifar100_amc.py --configs cifar100_amc/gs

# 99.5
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/cifar100_amc/gs.json
python cifar100_amc.py --configs cifar100_amc/gs

# 99
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/cifar100_amc/gs.json
python cifar100_amc.py --configs cifar100_amc/gs

sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/cifar100_amc/gs.json

# 97
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/cifar100_amc/gs.json
python cifar100_amc.py --configs cifar100_amc/gs

sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/cifar100_amc/gs.json

# 95
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/cifar100_amc/gs.json
python cifar100_amc.py --configs cifar100_amc/gs

# 90
sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 90.0/g' ./configs/cifar100_amc/gs.json
# python cifar100_amc.py --configs cifar100_amc/gs

# 80
sed -i 's/"gradient_sparse_rate": 90.0/"gradient_sparse_rate": 80.0/g' ./configs/cifar100_amc/gs.json
# python cifar100_amc.py --configs cifar100_amc/gs

sed -i 's/"gradient_sparse_rate": 80.0/"gradient_sparse_rate": 100.0/g' ./configs/cifar100_amc/gs.json
echo 'GS End'

#!/bin/bash
#SBATCH --job-name gs0_nsb              # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'GS90 80  dp  Begin'
# 100.0 #SBATCH --qos high
# python cifar_nsb.py --configs cifar_nsb/gs

# 99.5
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/cifar_nsb/gs.json
python cifar_nsb.py --configs cifar_nsb/gs

# 99
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/cifar_nsb/gs.json
python cifar_nsb.py --configs cifar_nsb/gs

sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/cifar_nsb/gs.json

# 97
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/cifar_nsb/gs.json
python cifar_nsb.py --configs cifar_nsb/gs

sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/cifar_nsb/gs.json

# 95
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/cifar_nsb/gs.json
python cifar_nsb.py --configs cifar_nsb/gs

# 90
sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 90.0/g' ./configs/cifar_nsb/gs.json
python cifar_nsb.py --configs cifar_nsb/gs

# 80
sed -i 's/"gradient_sparse_rate": 90.0/"gradient_sparse_rate": 80.0/g' ./configs/cifar_nsb/gs.json
python cifar_nsb.py --configs cifar_nsb/gs

sed -i 's/"gradient_sparse_rate": 80.0/"gradient_sparse_rate": 100.0/g' ./configs/cifar_nsb/gs.json
echo 'GS End'


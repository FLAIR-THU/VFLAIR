#!/bin/bash
#SBATCH --job-name dp0_aux             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'DP Begin'

# DP 0.0001
python cifar_nsb.py --configs cifar_nsb/ldp

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar_nsb/ldp.json
python cifar_nsb.py --configs cifar_nsb/ldp

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar_nsb/ldp.json
python cifar_nsb.py --configs cifar_nsb/ldp

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar_nsb/ldp.json
python cifar_nsb.py --configs cifar_nsb/ldp

# DP 0.1
sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar_nsb/ldp.json
python cifar_nsb.py --configs cifar_nsb/ldp

# DP 0.1
sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar_nsb/ldp.json
python cifar_nsb.py --configs cifar_nsb/ldp

sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar_nsb/ldp.json

echo 'DP End'



#!/bin/bash
#SBATCH --job-name dcor0_main            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# Begin with LaplaceDP 0.0001 
echo 'GDP agg nsb Begin'
# DP 0.0001
# python main_pipeline.py --configs test_cifar10_1 --gpu 0

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/test_cifar10_1.json
python main_pipeline.py --configs test_cifar10_1 --gpu 0

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/test_cifar10_1.json
python main_pipeline.py --configs test_cifar10_1 --gpu 0

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/test_cifar10_1.json
# python main_pipeline.py --configs test_cifar10_1 --gpu 0


sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/test_cifar10_1.json
echo 'GDP end'

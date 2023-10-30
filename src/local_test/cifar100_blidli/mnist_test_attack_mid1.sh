#!/bin/bash
#SBATCH --job-name mid0_main # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high
echo 'MID agg main begin' 

# 10000
echo '10000'
python cifar100_blidli.py --configs cifar100_blidli/mid1

sed -i 's/"lambda": 10000/"lambda": 100/g' ./configs/cifar100_blidli/mid1.json
python cifar100_blidli.py --configs cifar100_blidli/mid1

sed -i 's/"lambda": 100/"lambda": 1.0/g' ./configs/cifar100_blidli/mid1.json
python cifar100_blidli.py --configs cifar100_blidli/mid1

sed -i 's/"lambda": 1.0/"lambda": 0.0/g' ./configs/cifar100_blidli/mid1.json

echo 'MIDall end'




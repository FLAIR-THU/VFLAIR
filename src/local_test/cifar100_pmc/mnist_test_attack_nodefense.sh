#!/bin/bash
#SBATCH --job-name no0_pmc            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'No Defense cifar100 pmc Begin' #SBATCH --qos high

python cifar100_pmc.py --configs cifar100_pmc/nodefense

echo 'No Defense End'

#!/bin/bash
#SBATCH --job-name no0_main            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'No Defense Begin' #SBATCH --qos high

python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_nodefense

echo 'No Defense End'

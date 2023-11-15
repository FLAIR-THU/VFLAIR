#!/bin/bash
#SBATCH --job-name dp0_main             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'nuswide4 quant'

python main_pipeline_quant4.py --configs quantization/nuswide4

sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 5/g' ./configs/quantization/nuswide4.json

python main_pipeline_quant4.py --configs quantization/nuswide4

sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 1/g' ./configs/quantization/nuswide4.json
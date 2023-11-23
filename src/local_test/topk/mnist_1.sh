#!/bin/bash
#SBATCH --job-name dp0_main             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'mnist_1 topk'

### 0.9
python main_pipeline_topk.py --configs topk/mnist_1

sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 5/g' ./configs/topk/mnist_1.json
sed -i 's/"lr": 0.02/"lr": 0.01/g' ./configs/topk/mnist_1.json

python main_pipeline_topk.py --configs topk/mnist_1

sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 1/g' ./configs/topk/mnist_1.json
sed -i 's/"lr": 0.01/"lr": 0.02/g' ./configs/topk/mnist_1.json


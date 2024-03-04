#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'CoLA VMI_whitebox DP'

# 0.0001
python main_pipeline_llm.py --configs cola_mid

# 0.001
sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/cola_mid.json
python main_pipeline_llm.py --configs cola_mid

# 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/cola_mid.json
python main_pipeline_llm.py --configs cola_mid

# 0.05
sed -i 's/"lambda": 0.01/"lambda": 0.05/g' ./configs/cola_mid.json
python main_pipeline_llm.py --configs cola_mid

# 0.1
sed -i 's/"lambda": 0.05/"lambda": 0.1/g' ./configs/cola_mid.json
python main_pipeline_llm.py --configs cola_mid

# 0.5
sed -i 's/"lambda": 0.1/"lambda": 0.5/g' ./configs/cola_mid.json
python main_pipeline_llm.py --configs cola_mid



sed -i 's/"lambda": 0.5/"lambda": 0.0001/g' ./configs/cola_mid.json


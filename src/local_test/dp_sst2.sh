#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'CoLA VMI_whitebox'

# 1000
# python main_pipeline_llm.py --configs cola_dp

# 500
sed -i 's/"epsilon": 1000/"epsilon": 500/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 300
sed -i 's/"epsilon": 500/"epsilon": 300/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 100
sed -i 's/"epsilon": 300/"epsilon": 100/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 80
sed -i 's/"epsilon": 100/"epsilon": 80/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 50
sed -i 's/"epsilon": 80/"epsilon": 50/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 30
sed -i 's/"epsilon": 50/"epsilon": 30/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 10
sed -i 's/"epsilon": 30/"epsilon": 10/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 2
sed -i 's/"epsilon": 10/"epsilon": 2/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

# 0.5
sed -i 's/"epsilon": 2/"epsilon": 0.5/g' ./configs/cola_dp.json
python main_pipeline_llm.py --configs cola_dp

sed -i 's/"epsilon": 0.5/"epsilon": 1000/g' ./configs/cola_dp.json



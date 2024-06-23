#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'wo/dp SST2 3 encoders'

python main_pipeline_llm_3.py --seed 69 --configs sst2_wo

# 50
python main_pipeline_llm_3.py --seed 69 --configs sst2_dp

# 70
sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/sst2_dp.json
python main_pipeline_llm_3.py --seed 69 --configs sst2_dp

# 80
sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/sst2_dp.json
python main_pipeline_llm_3.py --seed 69 --configs sst2_dp

# 100
sed -i 's/"epsilon": 80/"epsilon": 100/g' ./configs/sst2_dp.json
python main_pipeline_llm_3.py --seed 69 --configs sst2_dp

# 500
sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/sst2_dp.json
python main_pipeline_llm_3.py --seed 69 --configs sst2_dp

sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_dp.json

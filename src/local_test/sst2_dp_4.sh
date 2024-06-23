#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'wo/dp SST2 4 encoders'
python main_pipeline_llm_4.py --seed 60 --configs sst2_wo_4
python main_pipeline_llm_4.py --seed 61 --configs sst2_wo_4

# 500
sed -i 's/"epsilon": 50/"epsilon": 500/g' ./configs/sst2_dp_4.json
python main_pipeline_llm_4.py --seed 61 --configs sst2_dp_4
sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_dp_4.json

for seed in {62,63,64,65,66,67,68,69}
    do
    python main_pipeline_llm_4.py --seed $seed --configs sst2_wo_4

    # 50
    python main_pipeline_llm_4.py --seed $seed --configs sst2_dp_4

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/sst2_dp_4.json
    python main_pipeline_llm_4.py --seed $seed --configs sst2_dp_4

    # 80
    sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/sst2_dp_4.json
    python main_pipeline_llm_4.py --seed $seed --configs sst2_dp_4

    # 100
    sed -i 's/"epsilon": 80/"epsilon": 100/g' ./configs/sst2_dp_4.json
    python main_pipeline_llm_4.py --seed $seed --configs sst2_dp_4

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/sst2_dp_4.json
    python main_pipeline_llm_4.py --seed $seed --configs sst2_dp_4

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_dp_4.json

done

#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'DP SST2 2Encoder'

for seed in {61,62,63,64,65}
    do

    python main_pipeline_llm.py --seed $seed --configs sst2_wo_2encoder

    # 50
    python main_pipeline_llm.py --seed $seed --configs sst2_dp_2encoder

    # 60
    sed -i 's/"epsilon": 50/"epsilon": 60/g' ./configs/sst2_dp_2encoder.json
    # python main_pipeline_llm.py --seed $seed --configs sst2_dp_2encoder

    # 70
    sed -i 's/"epsilon": 60/"epsilon": 70/g' ./configs/sst2_dp_2encoder.json
    python main_pipeline_llm.py --seed $seed --configs sst2_dp_2encoder

    # 80
    sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/sst2_dp_2encoder.json
    python main_pipeline_llm.py --seed $seed --configs sst2_dp_2encoder

    # 90
    sed -i 's/"epsilon": 80/"epsilon": 90/g' ./configs/sst2_dp_2encoder.json
    # python main_pipeline_llm.py --seed $seed --configs sst2_dp_2encoder

    # 100
    sed -i 's/"epsilon": 90/"epsilon": 100/g' ./configs/sst2_dp_2encoder.json
    python main_pipeline_llm.py --seed $seed --configs sst2_dp_2encoder

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/sst2_dp_2encoder.json
    python main_pipeline_llm.py --seed $seed --configs sst2_dp_2encoder

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_dp_2encoder.json
done
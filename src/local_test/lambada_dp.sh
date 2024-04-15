#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'DP SST2'

for seed in {65,66}
    do
    python main_pipeline_llm.py --seed $seed --configs lambada_wo

    # 50
    python main_pipeline_llm.py --seed $seed --configs lambada_dp

    # 60
    sed -i 's/"epsilon": 50/"epsilon": 60/g' ./configs/lambada_dp.json
    python main_pipeline_llm.py --seed $seed --configs lambada_dp

    # 70
    sed -i 's/"epsilon": 60/"epsilon": 70/g' ./configs/lambada_dp.json
    python main_pipeline_llm.py --seed $seed --configs lambada_dp

    # 80
    sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/lambada_dp.json
    python main_pipeline_llm.py --seed $seed --configs lambada_dp

    # 90
    sed -i 's/"epsilon": 80/"epsilon": 90/g' ./configs/lambada_dp.json
    python main_pipeline_llm.py --seed $seed --configs lambada_dp

    # 100
    sed -i 's/"epsilon": 90/"epsilon": 100/g' ./configs/lambada_dp.json
    python main_pipeline_llm.py --seed $seed --configs lambada_dp

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/lambada_dp.json
    python main_pipeline_llm.py --seed $seed --configs lambada_dp

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/lambada_dp.json
done
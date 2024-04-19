#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'MID Yelp'

for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm.py --seed $seed --configs yelp_wo

    ########## MIDModel_Linear ##########
    # 0.5
    python main_pipeline_llm.py --seed $seed --configs yelp_mid_es

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/yelp_mid_es.json
    python main_pipeline_llm.py --seed $seed --configs yelp_mid_es

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/yelp_mid_es.json
    python main_pipeline_llm.py --seed $seed --configs yelp_mid_es

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/yelp_mid_es.json
    python main_pipeline_llm.py --seed $seed --configs yelp_mid_es

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/yelp_mid_es.json
    python main_pipeline_llm.py --seed $seed --configs yelp_mid_es

    sed -i 's/"lambda": 0.0001/"lambda": 0.5/g' ./configs/yelp_mid_es.json
    # sed -i 's/"mid_model_name":"MIDModel_Linear"/"mid_model_name":"MIDModel_SqueezeLinear"/g' ./configs/yelp_mid_es.json

done

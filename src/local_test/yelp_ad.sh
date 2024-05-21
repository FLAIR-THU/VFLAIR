#!/bin/bash
#SBATCH --job-name yelp_ad_1             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high
                   

echo 'yelp 1 encoder Ad'


for seed in {62,63,64,65}
    do
    # 0.001
    python main_pipeline_llm_1.py --seed $seed --configs yelp_ad

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/yelp_ad.json
    python main_pipeline_llm_1.py --seed $seed --configs yelp_ad

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/yelp_ad.json
    python main_pipeline_llm_1.py --seed $seed --configs yelp_ad

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/yelp_ad.json
    python main_pipeline_llm_1.py --seed $seed --configs yelp_ad

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/yelp_ad.json
    python main_pipeline_llm_1.py --seed $seed --configs yelp_ad

    # 10
    sed -i 's/"lambda": 5.0/"lambda": 10.0/g' ./configs/yelp_ad.json
    # python main_pipeline_llm_1.py --seed $seed --configs yelp_ad


    sed -i 's/"lambda": 10.0/"lambda": 0.001/g' ./configs/yelp_ad.json

done

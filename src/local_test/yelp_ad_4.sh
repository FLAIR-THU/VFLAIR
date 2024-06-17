#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'yelp 4 encoder Ad'


# 0.001
# python main_pipeline_llm_4.py --seed 65 --configs yelp_ad_4

# 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/yelp_ad_4.json
# python main_pipeline_llm_4.py --seed 65 --configs yelp_ad_4

# 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/yelp_ad_4.json
python main_pipeline_llm_4.py --seed 65 --configs yelp_ad_4

# 1
sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/yelp_ad_4.json
python main_pipeline_llm_4.py --seed 65 --configs yelp_ad_4

# 5
sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/yelp_ad_4.json
python main_pipeline_llm_4.py --seed 65 --configs yelp_ad_4

sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/yelp_ad_4.json


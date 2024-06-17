#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'yelp 2 encoder Ad'


# 1.0
# WMI
sed -i 's/"lambda": 0.001/"lambda": 1.0/g' ./configs/yelp_ad_2_wmi.json
python main_pipeline_llm.py --seed 64 --configs yelp_ad_2_wmi
sed -i 's/"lambda": 1.0/"lambda": 0.001/g' ./configs/yelp_ad_2_wmi.json
# ALL
sed -i 's/"lambda": 0.001/"lambda": 1.0/g' ./configs/yelp_ad_2.json
python main_pipeline_llm.py --seed 65 --configs yelp_ad_2
sed -i 's/"lambda": 1.0/"lambda": 0.001/g' ./configs/yelp_ad_2.json


# 5.0
# WMI
sed -i 's/"lambda": 0.001/"lambda": 5.0/g' ./configs/yelp_ad_2_wmi.json
python main_pipeline_llm.py --seed 63 --configs yelp_ad_2_wmi
python main_pipeline_llm.py --seed 64 --configs yelp_ad_2_wmi
sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/yelp_ad_2_wmi.json
# ALL
sed -i 's/"lambda": 0.001/"lambda": 5.0/g' ./configs/yelp_ad_2.json
python main_pipeline_llm.py --seed 65 --configs yelp_ad_2
sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/yelp_ad_2.json


# 0.01
# WMI
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/yelp_ad_2_wmi.json
python main_pipeline_llm.py --seed 62 --configs yelp_ad_2_wmi
python main_pipeline_llm.py --seed 64 --configs yelp_ad_2_wmi
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/yelp_ad_2_wmi.json
# ALL
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/yelp_ad_2.json
python main_pipeline_llm.py --seed 65 --configs yelp_ad_2
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/yelp_ad_2.json


# 0.1
# WMI
sed -i 's/"lambda": 0.001/"lambda": 0.1/g' ./configs/yelp_ad_2_wmi.json
python main_pipeline_llm.py --seed 62 --configs yelp_ad_2_wmi
python main_pipeline_llm.py --seed 63 --configs yelp_ad_2_wmi
sed -i 's/"lambda": 0.1/"lambda": 0.001/g' ./configs/yelp_ad_2_wmi.json
# ALL
sed -i 's/"lambda": 0.001/"lambda": 0.1/g' ./configs/yelp_ad_2.json
python main_pipeline_llm.py --seed 64 --configs yelp_ad_2
python main_pipeline_llm.py --seed 65 --configs yelp_ad_2
sed -i 's/"lambda": 0.1/"lambda": 0.001/g' ./configs/yelp_ad_2.json


# # 10.0
# sed -i 's/"lambda": 0.001/"lambda": 10.0/g' ./configs/yelp_ad_2_wmi.json
# python main_pipeline_llm.py --seed 62 --configs yelp_ad_2_wmi
# python main_pipeline_llm.py --seed 63 --configs yelp_ad_2_wmi
# sed -i 's/"lambda": 10.0/"lambda": 0.001/g' ./configs/yelp_ad_2_wmi.json

# sed -i 's/"lambda": 0.001/"lambda": 10.0/g' ./configs/yelp_ad_2.json
# python main_pipeline_llm.py --seed 64 --configs yelp_ad_2
# python main_pipeline_llm.py --seed 65 --configs yelp_ad_2
# sed -i 's/"lambda": 10.0/"lambda": 0.001/g' ./configs/yelp_ad_2.json


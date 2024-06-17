#!/bin/bash
#SBATCH --job-name yelp_mid_3            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'MID 3 encoder Yelp'


########## MIDModel_Linear ##########
# 0.5
python main_pipeline_llm_3.py --seed 61 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 62 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 64 --configs yelp_mid_3


# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 61 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 63 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 64 --configs yelp_mid_3


# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 61 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 62 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 63 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 64 --configs yelp_mid_3


# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 60 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 61 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 62 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 63 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 64 --configs yelp_mid_3


# 0.0001
sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 60 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 61 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 62 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 63 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 64 --configs yelp_mid_3


# 0.00001
sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 60 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 61 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 62 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 63 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 64 --configs yelp_mid_3


# 0.05
sed -i 's/"lambda": 0.00001/"lambda": 0.05/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 61 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 62 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 64 --configs yelp_mid_3


# 0.005
sed -i 's/"lambda": 0.05/"lambda": 0.005/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 60 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 62 --configs yelp_mid_3
python main_pipeline_llm_3.py --seed 63 --configs yelp_mid_3

sed -i 's/"lambda": 0.005/"lambda": 0.5/g' ./configs/yelp_mid_3.json


### seed65
# 0.5
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

# 0.0001
sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

# 0.00001
sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

# 0.05
sed -i 's/"lambda": 0.00001/"lambda": 0.05/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

# 0.005
sed -i 's/"lambda": 0.05/"lambda": 0.005/g' ./configs/yelp_mid_3.json
python main_pipeline_llm_3.py --seed 65 --configs yelp_mid_3

sed -i 's/"lambda": 0.005/"lambda": 0.5/g' ./configs/yelp_mid_3.json


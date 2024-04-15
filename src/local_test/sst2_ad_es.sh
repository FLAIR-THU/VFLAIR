#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'SST2 Ad'


# 0.001
# python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

# 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/sst2_ad_es.json
# python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

# 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/sst2_ad_es.json
# python main_pipeline_llm.py --seed 61 --configs sst2_ad_es

# 0.5
sed -i 's/"lambda": 0.1/"lambda": 0.5/g' ./configs/sst2_ad_es.json
python main_pipeline_llm.py --seed 62 --configs sst2_ad_es

# 1
sed -i 's/"lambda": 0.5/"lambda": 1.0/g' ./configs/sst2_ad_es.json
python main_pipeline_llm.py --seed 61 --configs sst2_ad_es
python main_pipeline_llm.py --seed 62 --configs sst2_ad_es

# 5
sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/sst2_ad_es.json
python main_pipeline_llm.py --seed 62 --configs sst2_ad_es

sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/sst2_ad_es.json


for seed in {63,64,65}
    do
    # 0.001
    python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/sst2_ad_es.json
    python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/sst2_ad_es.json
    python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

    # 0.5
    sed -i 's/"lambda": 0.1/"lambda": 0.5/g' ./configs/sst2_ad_es.json
    python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

    # 1
    sed -i 's/"lambda": 0.5/"lambda": 1.0/g' ./configs/sst2_ad_es.json
    python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/sst2_ad_es.json
    python main_pipeline_llm.py --seed $seed --configs sst2_ad_es

    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/sst2_ad_es.json

done
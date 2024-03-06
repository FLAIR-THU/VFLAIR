#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'SST2 Ad'

# 0.5
python main_pipeline_llm.py --configs ad_sst2

# 0.2
sed -i 's/"lambda": 0.5/"lambda": 0.2/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 0.7
sed -i 's/"lambda": 0.2/"lambda": 0.7/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 1
sed -i 's/"lambda": 0.7/"lambda": 1.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 2
sed -i 's/"lambda": 1.0/"lambda": 2.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 5
sed -i 's/"lambda": 2.0/"lambda": 5.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 10
sed -i 's/"lambda": 5.0/"lambda": 10.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 50
sed -i 's/"lambda": 10.0/"lambda": 50.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 100
sed -i 's/"lambda": 50.0/"lambda": 100.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 200
sed -i 's/"lambda": 100.0/"lambda": 200.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 500
sed -i 's/"lambda": 200.0/"lambda": 500.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2

# 1000
sed -i 's/"lambda": 500.0/"lambda": 1000.0/g' ./configs/ad_sst2.json
python main_pipeline_llm.py --configs ad_sst2


sed -i 's/"lambda": 1000.0/"lambda": 0.5/g' ./configs/ad_sst2.json


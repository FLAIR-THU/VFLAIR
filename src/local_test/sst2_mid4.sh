#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'MID SST2'

########## MIDModelCNN_MaxUnpool2d ##########
# 0.5
# python main_pipeline_llm.py --configs sst2_mid4

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/sst2_mid4.json
python main_pipeline_llm.py --configs sst2_mid4

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/sst2_mid4.json
python main_pipeline_llm.py --configs sst2_mid4

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/sst2_mid4.json
python main_pipeline_llm.py --configs sst2_mid4

sed -i 's/"lambda": 0.001/"lambda": 0.5/g' ./configs/sst2_mid4.json




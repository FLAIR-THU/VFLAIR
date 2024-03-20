#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'MID SST2'

########## MIDModel_Linear ##########
# 0.5
python main_pipeline_llm.py --configs sst2_mid

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/sst2_mid.json
# python main_pipeline_llm.py --configs sst2_mid

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/sst2_mid.json
# python main_pipeline_llm.py --configs sst2_mid

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

sed -i 's/"lambda": 0.001/"lambda": 0.5/g' ./configs/sst2_mid.json

sed -i 's/"mid_model_name":"MIDModel_Linear"/"mid_model_name":"MIDModel_SqueezeLinear"/g' ./configs/sst2_mid.json

########## MIDModel_SqueezeLinear ##########
# 0.5
python main_pipeline_llm.py --configs sst2_mid

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/sst2_mid.json
# python main_pipeline_llm.py --configs sst2_mid

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/sst2_mid.json
# python main_pipeline_llm.py --configs sst2_mid

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

sed -i 's/"lambda": 0.001/"lambda": 0.5/g' ./configs/sst2_mid.json

sed -i 's/"mid_model_name":"MIDModel_SqueezeLinear"/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/g' ./configs/sst2_mid.json

########## MIDModelCNN_ConvTranspose2d ##########
# 0.5
python main_pipeline_llm.py --configs sst2_mid

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/sst2_mid.json
# python main_pipeline_llm.py --configs sst2_mid

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

sed -i 's/"lambda": 0.001/"lambda": 0.5/g' ./configs/sst2_mid.json

sed -i 's/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/"mid_model_name":"MIDModelCNN_MaxUnpool2d"/g' ./configs/sst2_mid.json


########## MIDModelCNN_MaxUnpool2d ##########
# 0.5
python main_pipeline_llm.py --configs sst2_mid

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

sed -i 's/"lambda": 0.001/"lambda": 0.5/g' ./configs/sst2_mid.json

sed -i 's/"mid_model_name":"MIDModelCNN_MaxUnpool2d"/"mid_model_name":"MIDModel_Linear"/g' ./configs/sst2_mid.json



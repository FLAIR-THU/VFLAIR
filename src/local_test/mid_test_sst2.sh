#!/bin/bash
#SBATCH --job-name SST2 mid             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'MID'

# MIDModel_Linear
python main_pipeline_llm.py --configs sst2_mid

# MIDModel_SqueezeLinear
sed -i 's/"mid_model_name":"MIDModel_Linear"/"mid_model_name":"MIDModel_SqueezeLinear"/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

# MIDModelCNN_ConvTranspose2d
sed -i 's/"mid_model_name":"MIDModel_SqueezeLinear"/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

# MIDModelCNN_MaxUnpool2d
sed -i 's/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/"mid_model_name":"MIDModelCNN_MaxUnpool2d"/g' ./configs/sst2_mid.json
python main_pipeline_llm.py --configs sst2_mid

sed -i 's/"mid_model_name":"MIDModelCNN_MaxUnpool2d"/"mid_model_name":"MIDModel_Linear"/g' ./configs/sst2_mid.json

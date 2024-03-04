#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'MID'

# MIDModel_Linear
python main_pipeline_llm.py --configs cola_mid1

# MIDModel_SqueezeLinear
sed -i 's/"mid_model_name":"MIDModel_Linear"/"mid_model_name":"MIDModel_SqueezeLinear"/g' ./configs/cola_mid1.json
python main_pipeline_llm.py --configs cola_mid1

# MIDModelCNN_ConvTranspose2d
sed -i 's/"mid_model_name":"MIDModel_SqueezeLinear"/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/g' ./configs/cola_mid1.json
python main_pipeline_llm.py --configs cola_mid1

# MIDModelCNN_MaxUnpool2d
sed -i 's/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/"mid_model_name":"MIDModelCNN_MaxUnpool2d"/g' ./configs/cola_mid1.json
python main_pipeline_llm.py --configs cola_mid1

sed -i 's/"mid_model_name":"MIDModelCNN_MaxUnpool2d"/"mid_model_name":"MIDModel_Linear"/g' ./configs/cola_mid1.json

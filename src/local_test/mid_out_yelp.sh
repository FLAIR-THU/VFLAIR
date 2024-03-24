#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'MID in test'

##### MIDModel_Linear
# 0.1
python main_pipeline_llm.py --configs yelp_out

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/yelp_out.json
python main_pipeline_llm.py --configs yelp_out

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/yelp_out.json
sed -i 's/"mid_model_name":"MIDModel_Linear"/"mid_model_name":"MIDModel_SqueezeLinear"/g' ./configs/yelp_out.json

##### MIDModel_SqueezeLinear
# 0.1
python main_pipeline_llm.py --configs yelp_out

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/yelp_out.json
python main_pipeline_llm.py --configs yelp_out

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/yelp_out.json
sed -i 's/"mid_model_name":"MIDModel_SqueezeLinear"/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/g' ./configs/yelp_out.json

##### MIDModelCNN_ConvTranspose2d
# 0.1
python main_pipeline_llm.py --configs yelp_out

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/yelp_out.json
python main_pipeline_llm.py --configs yelp_out

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/yelp_out.json
sed -i 's/"mid_model_name":"MIDModelCNN_ConvTranspose2d"/"mid_model_name":"MIDModel_SqueezeLinear"/g' ./configs/yelp_out.json


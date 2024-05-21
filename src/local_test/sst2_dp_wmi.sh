#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'dp SST2 WMI 1 encoders'



# 100
sed -i 's/"epsilon": 50/"epsilon": 100/g' ./configs/sst2_dp_wmi.json
python main_pipeline_llm_w.py --seed 64 --configs sst2_dp_wmi

# 500
sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/sst2_dp_wmi.json
python main_pipeline_llm_w.py --seed 64 --configs sst2_dp_wmi

sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_dp_wmi.json

for seed in {65,66,67,68,69}
    do

    # 50
    python main_pipeline_llm_w.py --seed $seed --configs sst2_dp_wmi

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/sst2_dp_wmi.json
    python main_pipeline_llm_w.py --seed $seed --configs sst2_dp_wmi

    # 80
    sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/sst2_dp_wmi.json
    python main_pipeline_llm_w.py --seed $seed --configs sst2_dp_wmi

    # 100
    sed -i 's/"epsilon": 80/"epsilon": 100/g' ./configs/sst2_dp_wmi.json
    python main_pipeline_llm_w.py --seed $seed --configs sst2_dp_wmi

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/sst2_dp_wmi.json
    python main_pipeline_llm_w.py --seed $seed --configs sst2_dp_wmi

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_dp_wmi.json

done

#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'CoLA VMI_whitebox'

# do not pad
python main_pipeline_llm_vmi.py --configs wmi_cola

# pad 50
sed -i 's/"padding": "do_not_pad"/"padding": "max_length"/g' ./configs/wmi_cola.json
python main_pipeline_llm_vmi.py --configs wmi_cola

# pad 50
sed -i 's/"max_length": 30/"max_length": 50/g' ./configs/wmi_cola.json
python main_pipeline_llm_vmi.py --configs wmi_cola

# pad 60
sed -i 's/"max_length": 50/"max_length": 60/g' ./configs/wmi_cola.json
python main_pipeline_llm_vmi.py --configs wmi_cola

# pad 70
sed -i 's/"max_length": 60/"max_length": 70/g' ./configs/wmi_cola.json
python main_pipeline_llm_vmi.py --configs wmi_cola

# pad 80
sed -i 's/"max_length": 70/"max_length": 80/g' ./configs/wmi_cola.json
python main_pipeline_llm_vmi.py --configs wmi_cola

# restore
sed -i 's/"max_length": 80/"max_length": 30/g' ./configs/wmi_cola.json
sed -i 's/"padding": "max_length"/"padding": "do_not_pad"/g' ./configs/wmi_cola.json






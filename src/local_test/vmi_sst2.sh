#!/bin/bash
#SBATCH --job-name SST2 VMI_whitebox           # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'SST2 VMI_whitebox'

# do not pad
# python main_pipeline_llm.py --configs vmi_sst2

# pad 30
sed -i 's/"padding": "do_not_pad"/"padding": "max_length"/g' ./configs/vmi_sst2.json
# python main_pipeline_llm.py --configs vmi_sst2

# pad 60
sed -i 's/"max_length": 30/"max_length": 60/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2

# pad 40
sed -i 's/"max_length": 60/"max_length": 65/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2

# pad 45
sed -i 's/"max_length": 65/"max_length": 70/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2

# pad 50
sed -i 's/"max_length": 75/"max_length": 80/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2

sed -i 's/"max_length": 80/"max_length": 30/g' ./configs/vmi_sst2.json


############## outside ################
sed -i 's/"padding_type": "inside"/"padding_type": "outside"/g' ./configs/vmi_sst2.json
sed -i 's/"padding": "do_not_pad"/"padding": "max_length"/g' ./configs/vmi_sst2.json
# python main_pipeline_llm.py --configs vmi_sst2

# pad 60
sed -i 's/"max_length": 30/"max_length": 60/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2

# pad 65
sed -i 's/"max_length": 60/"max_length": 65/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2

# pad 70
sed -i 's/"max_length": 65/"max_length": 70/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2

# pad 75
sed -i 's/"max_length": 75/"max_length": 80/g' ./configs/vmi_sst2.json
python main_pipeline_llm.py --configs vmi_sst2


sed -i 's/"max_length": 80/"max_length": 30/g' ./configs/vmi_sst2.json
sed -i 's/"padding_type": "outside"/"padding_type": "inside"/g' ./configs/vmi_sst2.json
sed -i 's/"padding": "max_length"/"padding": "do_not_pad"/g' ./configs/vmi_sst2.json






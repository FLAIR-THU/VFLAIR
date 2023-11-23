#!/bin/bash
#SBATCH --job-name no0_pmc            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'news20 agg Q1' #SBATCH --qos high 97 98 99
python main_basic_news.py --configs news20/basic_agg

#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'squad'

python main_pipeline_llm.py --configs squad_wo

python main_pipeline_llm.py --configs squad_dp

python main_pipeline_llm.py --configs squad_mid




#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'Lora/Vanilla finetune'

for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm_lora.py --seed $seed --configs lambada_gpt2_vanilla
    python main_pipeline_llm_lora.py --seed $seed --configs lambada_gpt2_lora

done

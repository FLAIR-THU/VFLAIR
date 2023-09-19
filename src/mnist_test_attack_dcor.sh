#!/bin/bash
#SBATCH --job-name dcor0_main            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# echo 'dCor main Begin '
# # Begin with 0.0001 #SBATCH --qos high
# # dCor 0.0001
# echo '0.0001'
# python main_pipeline.py --configs test_zty4_2dcor --gpu 1

# # dCor 0.001
# sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/test_zty4_2dcor.json
# # python main_pipeline.py --configs test_zty4_2dcor --gpu 1

# # dCor 0.01
# sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/test_zty4_2dcor.json
# python main_pipeline.py --configs test_zty4_2dcor --gpu 1

# # dCor 0.1
# sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/test_zty4_2dcor.json
# python main_pipeline.py --configs test_zty4_2dcor --gpu 1

# # dCor 0.3
# sed -i 's/"lambda": 0.1/"lambda": 0.3/g' ./configs/test_zty4_2dcor.json
# python main_pipeline.py --configs test_zty4_2dcor --gpu 1

# sed -i 's/"lambda": 0.3/"lambda": 0.0001/g' ./configs/test_zty4_2dcor.json

# echo 'dCor End'

# Begin with LaplaceDP 0.0001 
echo 'LDP agg nsb Begin'
# DP 0.0001
python main_pipeline.py --configs test_zty4_2ldp --gpu 1

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/test_zty4_2ldp.json
python main_pipeline.py --configs test_zty4_2ldp --gpu 1

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/test_zty4_2ldp.json
python main_pipeline.py --configs test_zty4_2ldp --gpu 1

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/test_zty4_2ldp.json
python main_pipeline.py --configs test_zty4_2ldp --gpu 1


sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/test_zty4_2ldp.json
echo 'LDP end'

#!/bin/bash
#SBATCH --job-name ldp0_nsds            # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high

# # Begin with LaplaceDP 0.0001 
# echo 'LDP agg nsds Begin'
# # DP 0.0001
# python main_pipeline.py --configs test_utk_dp --gpu 5

# # DP 0.001
# sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/test_utk_dp.json
# python main_pipeline.py --configs test_utk_dp --gpu 5

# # DP 0.01
# sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/test_utk_dp.json
# python main_pipeline.py --configs test_utk_dp --gpu 5

# # DP 0.1
# sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/test_utk_dp.json
# python main_pipeline.py --configs test_utk_dp --gpu 5


# sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/test_utk_dp.json


# # No Defense
# python main_pipeline.py --configs test_facescrub --gpu 5

# # Begin with LaplaceDP 0.0001 
# echo 'LDP agg nsds Begin'
# # DP 0.0001
# python main_pipeline.py --configs test_facescrub_dp --gpu 5

# # DP 0.001
# sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/test_facescrub_dp.json
# python main_pipeline.py --configs test_facescrub_dp --gpu 5

# # DP 0.01
# sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/test_facescrub_dp.json
# python main_pipeline.py --configs test_facescrub_dp --gpu 5

# # DP 0.1
# sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/test_facescrub_dp.json
# python main_pipeline.py --configs test_facescrub_dp --gpu 5


# sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/test_facescrub_dp.json


# # No Defense
# python main_pipeline.py --configs test_zty4_2dp --gpu 5

# Begin with LaplaceDP 0.0001 
echo 'LDP agg nsds Begin'
# DP 0.0001
python main_pipeline.py --configs test_zty4_2dp --gpu 6

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/test_zty4_2dp.json
python main_pipeline.py --configs test_zty4_2dp --gpu 6

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/test_zty4_2dp.json
python main_pipeline.py --configs test_zty4_2dp --gpu 6

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/test_zty4_2dp.json
python main_pipeline.py --configs test_zty4_2dp --gpu 6


sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/test_zty4_2dp.json
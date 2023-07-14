#!/bin/bash
#SBATCH --job-name gp0_main             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'GradPerturb main Begin'
# #SBATCH --qos high
# DP 0.01
python main_pipeline.py --configs main/mnist_test_attack_gp --gpu 6

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/main/mnist_test_attack_gp.json
python main_pipeline.py --configs main/mnist_test_attack_gp --gpu 6

# DP 0.3
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 0.3/g' ./configs/main/mnist_test_attack_gp.json
python main_pipeline.py --configs main/mnist_test_attack_gp --gpu 6

# DP 1.0
sed -i 's/"perturb_epsilon": 0.3/"perturb_epsilon": 1.0/g' ./configs/main/mnist_test_attack_gp.json
python main_pipeline.py --configs main/mnist_test_attack_gp --gpu 6

# DP 3.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 3.0/g' ./configs/main/mnist_test_attack_gp.json
python main_pipeline.py --configs main/mnist_test_attack_gp --gpu 6

# DP 10.0
sed -i 's/"perturb_epsilon": 3.0/"perturb_epsilon": 10.0/g' ./configs/main/mnist_test_attack_gp.json
python main_pipeline.py --configs main/mnist_test_attack_gp --gpu 6

sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 0.01/g' ./configs/main/mnist_test_attack_gp.json

echo 'GradPerturb End'


# echo 'GradPerturb main Begin'
# # #SBATCH --qos high
# # DP 0.01
# # python main_pipeline.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 0.1
# sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_pipeline.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 0.3
# sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 0.3/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_pipeline.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 1.0
# sed -i 's/"perturb_epsilon": 0.3/"perturb_epsilon": 1.0/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_pipeline.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 3.0
# sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 3.0/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_pipeline.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 10.0
# sed -i 's/"perturb_epsilon": 3.0/"perturb_epsilon": 10.0/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_pipeline.py --configs main_split/mnist_test_attack_gp --gpu 1

# sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 0.01/g' ./configs/main_split/mnist_test_attack_gp.json

# echo 'GradPerturb End'



echo 'dCor main Begin '
# Begin with 0.0001 #SBATCH --qos high
# dCor 0.0001
echo '0.0001'
python main_pipeline.py --configs main/mnist_test_attack_dcor --gpu 6

# dCor 0.001
sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/main/mnist_test_attack_dcor.json
python main_pipeline.py --configs main/mnist_test_attack_dcor --gpu 6

# dCor 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/main/mnist_test_attack_dcor.json
python main_pipeline.py --configs main/mnist_test_attack_dcor --gpu 6

# dCor 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/main/mnist_test_attack_dcor.json
python main_pipeline.py --configs main/mnist_test_attack_dcor --gpu 6

# dCor 0.3
sed -i 's/"lambda": 0.1/"lambda": 0.3/g' ./configs/main/mnist_test_attack_dcor.json
python main_pipeline.py --configs main/mnist_test_attack_dcor --gpu 6

sed -i 's/"lambda": 0.3/"lambda": 0.0001/g' ./configs/main/mnist_test_attack_dcor.json

echo 'dCor End'


echo 'GS main latest Begin'
# 100.0 #SBATCH --qos high
# python main_pipeline.py --configs main/mnist_test_attack_gs --gpu 6

# 99.5
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/main/mnist_test_attack_gs.json
python main_pipeline.py --configs main/mnist_test_attack_gs --gpu 6

# 99
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/main/mnist_test_attack_gs.json
python main_pipeline.py --configs main/mnist_test_attack_gs --gpu 6

sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/main/mnist_test_attack_gs.json
# python main_pipeline.py --configs main/mnist_test_attack_gs --gpu 6

# 97
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/main/mnist_test_attack_gs.json
python main_pipeline.py --configs main/mnist_test_attack_gs --gpu 6

sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/main/mnist_test_attack_gs.json
# python main_pipeline.py --configs main/mnist_test_attack_gs --gpu 6

# 95
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/main/mnist_test_attack_gs.json
python main_pipeline.py --configs main/mnist_test_attack_gs --gpu 6

sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 100.0/g' ./configs/main/mnist_test_attack_gs.json
echo 'GS End'
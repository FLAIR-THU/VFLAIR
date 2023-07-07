#!/bin/bash
#SBATCH --job-name gp0_main             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'GradPerturb main Begin'
# #SBATCH --qos high
# DP 0.01
# python main_separate.py --configs main/mnist_test_attack_gp --gpu 1

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/main/mnist_test_attack_gp.json
# python main_separate.py --configs main/mnist_test_attack_gp --gpu 1

# DP 0.3
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 0.3/g' ./configs/main/mnist_test_attack_gp.json
python main_separate.py --configs main/mnist_test_attack_gp --gpu 1

# DP 1.0
sed -i 's/"perturb_epsilon": 0.3/"perturb_epsilon": 1.0/g' ./configs/main/mnist_test_attack_gp.json
# python main_separate.py --configs main/mnist_test_attack_gp --gpu 1

# DP 3.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 3.0/g' ./configs/main/mnist_test_attack_gp.json
# python main_separate.py --configs main/mnist_test_attack_gp --gpu 1

# DP 10.0
sed -i 's/"perturb_epsilon": 3.0/"perturb_epsilon": 10.0/g' ./configs/main/mnist_test_attack_gp.json
# python main_separate.py --configs main/mnist_test_attack_gp --gpu 1

sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 0.01/g' ./configs/main/mnist_test_attack_gp.json

echo 'GradPerturb End'


# echo 'GradPerturb main Begin'
# # #SBATCH --qos high
# # DP 0.01
# # python main_separate.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 0.1
# sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_separate.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 0.3
# sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 0.3/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_separate.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 1.0
# sed -i 's/"perturb_epsilon": 0.3/"perturb_epsilon": 1.0/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_separate.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 3.0
# sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 3.0/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_separate.py --configs main_split/mnist_test_attack_gp --gpu 1

# # DP 10.0
# sed -i 's/"perturb_epsilon": 3.0/"perturb_epsilon": 10.0/g' ./configs/main_split/mnist_test_attack_gp.json
# # python main_separate.py --configs main_split/mnist_test_attack_gp --gpu 1

# sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 0.01/g' ./configs/main_split/mnist_test_attack_gp.json

# echo 'GradPerturb End'
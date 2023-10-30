#!/bin/bash
#SBATCH --job-name gp10_main             # 任务名叫 example
#SBATCH --gres gp1u:a100:1                   # 每个子任务都用一张 A100 gp1U
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


echo 'GradPerturb main Begin'
# #SBATCH --qos high
# DP 0.01
# python cifar100_blidli.py --configs cifar100_blidli/gp1

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/cifar100_blidli/gp1.json
# python cifar100_blidli.py --configs cifar100_blidli/gp1

# DP 1.0
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 1.0/g' ./configs/cifar100_blidli/gp1.json
# python cifar100_blidli.py --configs cifar100_blidli/gp1

# DP 10.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 10.0/g' ./configs/cifar100_blidli/gp1.json
echo 'gp 10'
python cifar100_blidli.py --configs cifar100_blidli/gp1

sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 0.01/g' ./configs/cifar100_blidli/gp1.json

echo 'GradPerturb End'

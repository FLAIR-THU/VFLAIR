#!/bin/bash
#SBATCH --job-name dp0_main             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'DP/LDP Add Begin'

# DP 0.0001
# python cifar100_blidli.py --configs cifar100_blidli/dp
# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar100_blidli/dp.json
# python cifar100_blidli.py --configs cifar100_blidli/dp
# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar100_blidli/dp.json
# python cifar100_blidli.py --configs cifar100_blidli/dp
# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar100_blidli/dp.json
# python cifar100_blidli.py --configs cifar100_blidli/dp
echo 'add'
# DP 0.00001
sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar100_blidli/dp.json
python cifar100_blidli.py --configs cifar100_blidli/dp
# DP 0.000001
sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar100_blidli/dp.json
python cifar100_blidli.py --configs cifar100_blidli/dp
sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar100_blidli/dp.json
echo 'DP End'

echo 'LDP  Begin'
# DP 0.0001
# python cifar100_blidli.py --configs cifar100_blidli/ldp
# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar100_blidli/ldp.json
# python cifar100_blidli.py --configs cifar100_blidli/ldp
# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar100_blidli/ldp.json
# python cifar100_blidli.py --configs cifar100_blidli/ldp
# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar100_blidli/ldp.json
# python cifar100_blidli.py --configs cifar100_blidli/ldp
echo 'add'
# DP 0.00001
sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar100_blidli/ldp.json
python cifar100_blidli.py --configs cifar100_blidli/ldp
# DP 0.000001
sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar100_blidli/ldp.json
python cifar100_blidli.py --configs cifar100_blidli/ldp
sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar100_blidli/ldp.json
echo 'LDP End'

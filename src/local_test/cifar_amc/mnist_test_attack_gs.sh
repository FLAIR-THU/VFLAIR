#!/bin/bash
#SBATCH --job-name gs0_amc              # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'GS90 80  dp   gp Begin'
# 100.0 #SBATCH --qos high
# python cifar_amc.py --configs cifar_amc/gs

# 99.5
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/cifar_amc/gs.json
# python cifar_amc.py --configs cifar_amc/gs

# 99
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/cifar_amc/gs.json
# python cifar_amc.py --configs cifar_amc/gs

sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/cifar_amc/gs.json

# 97
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/cifar_amc/gs.json
# python cifar_amc.py --configs cifar_amc/gs

sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/cifar_amc/gs.json

# 95
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/cifar_amc/gs.json
# python cifar_amc.py --configs cifar_amc/gs

# 90
sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 90.0/g' ./configs/cifar_amc/gs.json
python cifar_amc.py --configs cifar_amc/gs

# 80
sed -i 's/"gradient_sparse_rate": 90.0/"gradient_sparse_rate": 80.0/g' ./configs/cifar_amc/gs.json
python cifar_amc.py --configs cifar_amc/gs

sed -i 's/"gradient_sparse_rate": 80.0/"gradient_sparse_rate": 100.0/g' ./configs/cifar_amc/gs.json
echo 'GS End'


echo 'DP Begin'

# DP 0.0001
# python cifar_amc.py --configs cifar_amc/dp

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar_amc/dp.json
# python cifar_amc.py --configs cifar_amc/dp

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar_amc/dp.json
# python cifar_amc.py --configs cifar_amc/dp

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar_amc/dp.json
echo 'DP 0.1'
# python cifar_amc.py --configs cifar_amc/dp

sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar_amc/dp.json
python cifar_amc.py --configs cifar_amc/dp

sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar_amc/dp.json
python cifar_amc.py --configs cifar_amc/dp

sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar_amc/dp.json

echo 'DP End'

#!/bin/bash
#SBATCH --job-name dp0_amc             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'LDP Begin'

# DP 0.0001
# python cifar_amc.py --configs cifar_amc/ldp

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar_amc/ldp.json
# python cifar_amc.py --configs cifar_amc/ldp

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar_amc/ldp.json
# python cifar_amc.py --configs cifar_amc/ldp

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar_amc/ldp.json
echo 'DP 0.1'
# python cifar_amc.py --configs cifar_amc/ldp

sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar_amc/ldp.json
python cifar_amc.py --configs cifar_amc/ldp

sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar_amc/ldp.json
python cifar_amc.py --configs cifar_amc/ldp

sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar_amc/ldp.json

echo 'LDP End'


echo 'GradPerturb amc Begin'
# #SBATCH --qos high
# DP 0.01
# python cifar_amc.py --configs cifar_amc/gp

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/cifar_amc/gp.json
# python cifar_amc.py --configs cifar_amc/gp

# DP 1.0
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 1.0/g' ./configs/cifar_amc/gp.json
# python cifar_amc.py --configs cifar_amc/gp

# DP 10.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 10.0/g' ./configs/cifar_amc/gp.json
# python cifar_amc.py --configs cifar_amc/gp

# DP 50.0
sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 50.0/g' ./configs/cifar_amc/gp.json
python cifar_amc.py --configs cifar_amc/gp

# DP 10.0
sed -i 's/"perturb_epsilon": 50.0/"perturb_epsilon": 100.0/g' ./configs/cifar_amc/gp.json
python cifar_amc.py --configs cifar_amc/gp

sed -i 's/"perturb_epsilon": 100.0/"perturb_epsilon": 0.01/g' ./configs/cifar_amc/gp.json

echo 'GradPerturb End'


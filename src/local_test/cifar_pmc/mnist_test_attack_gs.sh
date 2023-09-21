#!/bin/bash
#SBATCH --job-name gs0_pmc              # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

echo 'GS90 80  dp   gp Begin'
# 100.0 #SBATCH --qos high
# python cifar_pmc.py --configs cifar_pmc/gs

# 99.5
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/cifar_pmc/gs.json
# python cifar_pmc.py --configs cifar_pmc/gs

# 99
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/cifar_pmc/gs.json
# python cifar_pmc.py --configs cifar_pmc/gs

sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/cifar_pmc/gs.json

# 97
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/cifar_pmc/gs.json
# python cifar_pmc.py --configs cifar_pmc/gs

sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/cifar_pmc/gs.json

# 95
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/cifar_pmc/gs.json
# python cifar_pmc.py --configs cifar_pmc/gs

# 90
sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 90.0/g' ./configs/cifar_pmc/gs.json
python cifar_pmc.py --configs cifar_pmc/gs

# 80
sed -i 's/"gradient_sparse_rate": 90.0/"gradient_sparse_rate": 80.0/g' ./configs/cifar_pmc/gs.json
python cifar_pmc.py --configs cifar_pmc/gs

sed -i 's/"gradient_sparse_rate": 80.0/"gradient_sparse_rate": 100.0/g' ./configs/cifar_pmc/gs.json
echo 'GS End'


echo 'DP Begin'

# DP 0.0001
# python cifar_pmc.py --configs cifar_pmc/dp

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar_pmc/dp.json
# python cifar_pmc.py --configs cifar_pmc/dp

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar_pmc/dp.json
# python cifar_pmc.py --configs cifar_pmc/dp

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar_pmc/dp.json
echo 'DP 0.1'
# python cifar_pmc.py --configs cifar_pmc/dp

sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar_pmc/dp.json
python cifar_pmc.py --configs cifar_pmc/dp

sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar_pmc/dp.json
python cifar_pmc.py --configs cifar_pmc/dp

sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar_pmc/dp.json

echo 'DP End'

#!/bin/bash
#SBATCH --job-name dp0_pmc             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'LDP Begin'

# DP 0.0001
# python cifar_pmc.py --configs cifar_pmc/ldp

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar_pmc/ldp.json
# python cifar_pmc.py --configs cifar_pmc/ldp

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar_pmc/ldp.json
# python cifar_pmc.py --configs cifar_pmc/ldp

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar_pmc/ldp.json
echo 'DP 0.1'
# python cifar_pmc.py --configs cifar_pmc/ldp

sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar_pmc/ldp.json
python cifar_pmc.py --configs cifar_pmc/ldp

sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar_pmc/ldp.json
python cifar_pmc.py --configs cifar_pmc/ldp

sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar_pmc/ldp.json

echo 'LDP End'


echo 'GradPerturb pmc Begin'
# #SBATCH --qos high
# DP 0.01
# python cifar_pmc.py --configs cifar_pmc/gp

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/cifar_pmc/gp.json
# python cifar_pmc.py --configs cifar_pmc/gp

# DP 1.0
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 1.0/g' ./configs/cifar_pmc/gp.json
# python cifar_pmc.py --configs cifar_pmc/gp

# DP 10.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 10.0/g' ./configs/cifar_pmc/gp.json
# python cifar_pmc.py --configs cifar_pmc/gp

# DP 50.0
sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 50.0/g' ./configs/cifar_pmc/gp.json
python cifar_pmc.py --configs cifar_pmc/gp

# DP 10.0
sed -i 's/"perturb_epsilon": 50.0/"perturb_epsilon": 100.0/g' ./configs/cifar_pmc/gp.json
python cifar_pmc.py --configs cifar_pmc/gp

sed -i 's/"perturb_epsilon": 100.0/"perturb_epsilon": 0.01/g' ./configs/cifar_pmc/gp.json

echo 'GradPerturb End'


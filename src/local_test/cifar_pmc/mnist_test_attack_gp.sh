
echo 'GradPerturb bu pmc Begin'
# #SBATCH --qos high
# DP 0.01
python cifar_pmc.py --configs cifar_pmc/gp1

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/cifar_pmc/gp1.json
python cifar_pmc.py --configs cifar_pmc/gp1

# DP 1.0
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 1.0/g' ./configs/cifar_pmc/gp1.json
python cifar_pmc.py --configs cifar_pmc/gp1

# DP 10.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 10.0/g' ./configs/cifar_pmc/gp1.json
python cifar_pmc.py --configs cifar_pmc/gp1

# DP 50.0
sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 50.0/g' ./configs/cifar_pmc/gp1.json
# python cifar_pmc.py --configs cifar_pmc/gp1

# DP 10.0
sed -i 's/"perturb_epsilon": 50.0/"perturb_epsilon": 100.0/g' ./configs/cifar_pmc/gp1.json
# python cifar_pmc.py --configs cifar_pmc/gp1

sed -i 's/"perturb_epsilon": 100.0/"perturb_epsilon": 0.01/g' ./configs/cifar_pmc/gp1.json

echo 'GradPerturb End'


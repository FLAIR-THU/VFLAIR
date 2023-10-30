
echo 'DP Begin'

# DP 0.0001
python cifar100_amc.py --configs cifar100_amc/dp

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/cifar100_amc/dp.json
python cifar100_amc.py --configs cifar100_amc/dp

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/cifar100_amc/dp.json
python cifar100_amc.py --configs cifar100_amc/dp

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/cifar100_amc/dp.json
echo 'DP 0.1'
python cifar100_amc.py --configs cifar100_amc/dp

sed -i 's/"dp_strength": 0.1/"dp_strength": 0.00001/g' ./configs/cifar100_amc/dp.json
# python cifar100_amc.py --configs cifar100_amc/dp

sed -i 's/"dp_strength": 0.00001/"dp_strength": 0.000001/g' ./configs/cifar100_amc/dp.json
# python cifar100_amc.py --configs cifar100_amc/dp

sed -i 's/"dp_strength": 0.000001/"dp_strength": 0.0001/g' ./configs/cifar100_amc/dp.json

echo 'DP End'

echo 'MID cifar100 grn begin' 
echo '0.0'
python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 0.0/"lambda": 1e-8/g' ./configs/cifar100_grn_mid/mid.json
python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 1e-8/"lambda": 1e-6/g' ./configs/cifar100_grn_mid/mid.json
python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 1e-6/"lambda": 0.0001/g' ./configs/cifar100_grn_mid/mid.json
# python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 0.0001/"lambda": 0.01/g' ./configs/cifar100_grn_mid/mid.json
# python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/cifar100_grn_mid/mid.json
# python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/cifar100_grn_mid/mid.json
# python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 1.0/"lambda": 100/g' ./configs/cifar100_grn_mid/mid.json
# python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 100/"lambda": 10000/g' ./configs/cifar100_grn_mid/mid.json
# python cifar100_grn_mid.py --configs cifar100_grn_mid/mid

sed -i 's/"lambda": 10000/"lambda": 0.0/g' ./configs/cifar100_grn_mid/mid.json

echo 'MIDall end'




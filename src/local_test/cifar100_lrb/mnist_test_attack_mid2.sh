#!/bin/bash
#SBATCH --job-name mid0_lrb # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high
echo 'MID cifar100 bd begin' 

# python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 0.0/"lambda": 1e-8/g' ./configs/cifar100_lrb/mid2.json
# python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 1e-8/"lambda": 1e-6/g' ./configs/cifar100_lrb/mid2.json
# python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 1e-6/"lambda": 0.0001/g' ./configs/cifar100_lrb/mid2.json
# python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 0.0001/"lambda": 0.01/g' ./configs/cifar100_lrb/mid2.json
# python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/cifar100_lrb/mid2.json
echo '0.1'
# python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/cifar100_lrb/mid2.json
echo '1.0'
python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 1.0/"lambda": 100/g' ./configs/cifar100_lrb/mid2.json
echo '100'
python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 100/"lambda": 10000/g' ./configs/cifar100_lrb/mid2.json
echo '10000'
python cifar100_lrb.py --configs cifar100_lrb/mid2

sed -i 's/"lambda": 10000/"lambda": 0.0/g' ./configs/cifar100_lrb/mid2.json

echo 'MIDall end'




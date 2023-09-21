#!/bin/bash
#SBATCH --job-name mid0_main # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high
echo 'MID  fr begin' 

# python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 0.0/"lambda": 1e-8/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
# python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 1e-8/"lambda": 1e-6/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
# python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 1e-6/"lambda": 0.0001/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
# python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 0.0001/"lambda": 0.01/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
# python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
echo '0.1'
python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 1.0/"lambda": 100/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 100/"lambda": 10000/g' ./configs/mnist_grn/mnist_test_attack_mid1.json
python main_pipeline_grn.py --configs mnist_grn/mnist_test_attack_mid1

sed -i 's/"lambda": 10000/"lambda": 0.0/g' ./configs/mnist_grn/mnist_test_attack_mid1.json

echo 'MIDall end'



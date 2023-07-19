# #!/bin/bash
# #SBATCH --job-name no0_nsds            # 任务名叫 example
# #SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
# #SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# echo 'No Defense agg Begin' #SBATCH --qos high

# python main_pipeline.py --configs nsds/mnist_test_attack_nodefense --gpu 7

# echo 'No Defense End'


# #!/bin/bash
# #SBATCH --job-name mid0_nsds # 任务名叫 example
# #SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
# #SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
# #SBATCH --qos high
# echo 'MID agg nsds begin'

# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 0.0/"lambda": 1e-8/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 1e-8/"lambda": 1e-6/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 1e-6/"lambda": 0.0001/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 0.0001/"lambda": 0.01/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 1.0/"lambda": 100/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 100/"lambda": 10000/g' ./configs/nsds/mnist_test_attack_mid.json
# python main_pipeline.py --configs nsds/mnist_test_attack_mid --gpu 7

# sed -i 's/"lambda": 10000/"lambda": 0.0/g' ./configs/nsds/mnist_test_attack_mid.json

# echo 'MIDall end'



# #!/bin/bash
# #SBATCH --job-name ldp0_nsds            # 任务名叫 example
# #SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
# #SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
# #SBATCH --qos high

# # Begin with LaplaceDP 0.0001 
# echo 'LDP agg nsds Begin'
# # DP 0.0001
# python main_pipeline.py --configs nsds/mnist_test_attack_ldp --gpu 7

# # DP 0.001
# sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/nsds/mnist_test_attack_ldp.json
# python main_pipeline.py --configs nsds/mnist_test_attack_ldp --gpu 7

# # DP 0.01
# sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/nsds/mnist_test_attack_ldp.json
# python main_pipeline.py --configs nsds/mnist_test_attack_ldp --gpu 7

# # DP 0.1
# sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/nsds/mnist_test_attack_ldp.json
# python main_pipeline.py --configs nsds/mnist_test_attack_ldp --gpu 7


# sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/nsds/mnist_test_attack_ldp.json

# #!/bin/bash
# #SBATCH --job-name gs0_nsds              # 任务名叫 example
# #SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
# #SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

# echo 'GS agg nsds latest Begin'
# # 100.0 #SBATCH --qos high
# # python main_pipeline.py --configs nsds/mnist_test_attack_gs --gpu 7

# # 99.5
# sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/nsds/mnist_test_attack_gs.json
# python main_pipeline.py --configs nsds/mnist_test_attack_gs --gpu 7

# # 99
# sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/nsds/mnist_test_attack_gs.json
# python main_pipeline.py --configs nsds/mnist_test_attack_gs --gpu 7

# sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/nsds/mnist_test_attack_gs.json
# # python main_pipeline.py --configs nsds/mnist_test_attack_gs --gpu 7

# # 97
# sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/nsds/mnist_test_attack_gs.json
# python main_pipeline.py --configs nsds/mnist_test_attack_gs --gpu 7

# sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/nsds/mnist_test_attack_gs.json
# # python main_pipeline.py --configs nsds/mnist_test_attack_gs --gpu 7

# # 95
# sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/nsds/mnist_test_attack_gs.json
# python main_pipeline.py --configs nsds/mnist_test_attack_gs --gpu 7

# sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 100.0/g' ./configs/nsds/mnist_test_attack_gs.json
# echo 'GS End'


#!/bin/bash
#SBATCH --job-name mid0_nsds # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --qos high
echo 'MID agg nsds begin'

python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 0.0/"lambda": 1e-8/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 1e-8/"lambda": 1e-6/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 1e-6/"lambda": 0.0001/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 0.0001/"lambda": 0.01/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 1.0/"lambda": 100/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 100/"lambda": 10000/g' ./configs/nsds/mnist_test_attack_mid.json
python main_separate2.py --configs nsds/mnist_test_attack_mid --gpu 6

sed -i 's/"lambda": 10000/"lambda": 0.0/g' ./configs/nsds/mnist_test_attack_mid.json

echo 'MIDall end'
#!/bin/bash
#SBATCH --job-name mid0_main # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# echo 'Adversarial begin' #SBATCH --qos high

# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/test_utk_adv.json
# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/test_utk_adv.json
# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/test_utk_adv.json
# echo '0.1'
# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/test_utk_adv.json
# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 1.0/"lambda": 10/g' ./configs/test_utk_adv.json
# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 10/"lambda": 100/g' ./configs/test_utk_adv.json
# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 100/"lambda": 10000/g' ./configs/test_utk_adv.json
# python main_pipeline.py --configs test_utk_adv --gpu 7

# sed -i 's/"lambda": 10000/"lambda": 0.0/g' ./configs/test_utk_adv.json

# echo 'Adversarial end'



echo 'Adversarial begin' #SBATCH --qos high

python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 0.0/"lambda": 0.0001/g' ./configs/test_facescrub_adv.json
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/test_facescrub_adv.json
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/test_facescrub_adv.json
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/test_facescrub_adv.json
echo '0.1'
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/test_facescrub_adv.json
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 1.0/"lambda": 10/g' ./configs/test_facescrub_adv.json
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 10/"lambda": 100/g' ./configs/test_facescrub_adv.json
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 100/"lambda": 10000/g' ./configs/test_facescrub_adv.json
python main_pipeline.py --configs test_facescrub_adv --gpu 7

sed -i 's/"lambda": 10000/"lambda": 0.0/g' ./configs/test_facescrub_adv.json

echo 'Adversarial end'
# #!/bin/bash
# # README: run this file under src folder
# # for MID+bli, mai_lr fix to 0.05; w/o top_model, mid_lr changed 0.001/0.1 for w/o top model;
# # Noticl: mid applying party should be manually changed and rerun this script

# # ###### make sure starts with non-trainable head ######
# sed -i 's/"lr": 0.1/"lr": 0.001/g' ./configs/test/mnist_bli_mid_configs.json
# sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_bli_mid_configs.json
# sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 0.0/"lambda": 1e-9/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-9/"lambda": 1e-8/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-8/"lambda": 1e-7/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-7/"lambda": 1e-6/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-6/"lambda": 1e-5/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-5/"lambda": 0.0001/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/test/mnist_bli_mid_configs.json


# # ###### change to trainable head ######
# sed -i 's/"lr": 0.001/"lr": 0.1/g' ./configs/test/mnist_bli_mid_configs.json
# sed -i 's/"apply_trainable_layer": 0/"apply_trainable_layer": 1/g' ./configs/test/mnist_bli_mid_configs.json
# sed -i 's/"global_model": "ClassificationModelHostHead"/"global_model": "ClassificationModelHostTrainableHead"/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 0.0/"lambda": 1e-9/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-9/"lambda": 1e-8/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-8/"lambda": 1e-7/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-7/"lambda": 1e-6/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-6/"lambda": 1e-5/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
# sed -i 's/"lambda": 1e-5/"lambda": 0.0001/g' ./configs/test/mnist_bli_mid_configs.json
# # for i in `seq 91 100`; do 
# for i in `seq 97 97`; do 
#     python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
# done
sed -i 's/"lambda": 0.0001/"lambda": 0.001/g' ./configs/test/mnist_bli_mid_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
done
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/test/mnist_bli_mid_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
done
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/test/mnist_bli_mid_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_bli_mid_configs --seed $i
done
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/test/mnist_bli_mid_configs.json

sed -i 's/"lr": 0.1/"lr": 0.001/g' ./configs/test/mnist_bli_mid_configs.json
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_bli_mid_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_bli_mid_configs.json

# srun --gres=gpu:a100-80G:1 --time=1-00:00:00 --mem=20000 bash ./local_test/mnist_bli_gs.sh
#!/bin/bash
# README: run this file under src folder

# ###### make sure starts with non-trainable head ######
sed -i 's/"name": "GaussianDP"/"name": "LaplaceDP"/g' ./configs/test/mnist_backdoor_dp_configs.json
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_backdoor_dp_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.0005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0005/"dp_strength": 0.001/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
ed -i 's/"dp_strength": 0.001/"dp_strength": 0.005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.005/"dp_strength": 0.01/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.05/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.05/"dp_strength": 0.1/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.1/"dp_strength": 0.5/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.5/"dp_strength": 0.0001/g' ./configs/test/mnist_backdoor_dp_configs.json


# ###### change to trainable head ######
sed -i 's/"apply_trainable_layer": 0/"apply_trainable_layer": 1/g' ./configs/test/mnist_backdoor_dp_configs.json
sed -i 's/"global_model": "ClassificationModelHostHead"/"global_model": "ClassificationModelHostTrainableHead"/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.0005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0005/"dp_strength": 0.001/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.005/"dp_strength": 0.01/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.05/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.05/"dp_strength": 0.1/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.1/"dp_strength": 0.5/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.5/"dp_strength": 0.0001/g' ./configs/test/mnist_backdoor_dp_configs.json


# ###### change to gaussian-dp and repeat ######
sed -i 's/"name": "LaplaceDP"/"name": "GaussianDP"/g' ./configs/test/mnist_backdoor_dp_configs.json
# ###### make sure starts with non-trainable head ######
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_backdoor_dp_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.0005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0005/"dp_strength": 0.001/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.005/"dp_strength": 0.01/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.05/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.05/"dp_strength": 0.1/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.1/"dp_strength": 0.5/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.5/"dp_strength": 0.0001/g' ./configs/test/mnist_backdoor_dp_configs.json


# ###### change to trainable head ######
sed -i 's/"apply_trainable_layer": 0/"apply_trainable_layer": 1/g' ./configs/test/mnist_backdoor_dp_configs.json
sed -i 's/"global_model": "ClassificationModelHostHead"/"global_model": "ClassificationModelHostTrainableHead"/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.0005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.0005/"dp_strength": 0.001/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.005/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.005/"dp_strength": 0.01/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.05/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.05/"dp_strength": 0.1/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.1/"dp_strength": 0.5/g' ./configs/test/mnist_backdoor_dp_configs.json
# for i in `seq 91 100`; do 
for i in `seq 97 97`; do 
    python main_separate.py --configs test/mnist_backdoor_dp_configs --seed $i
done
sed -i 's/"dp_strength": 0.5/"dp_strength": 0.0001/g' ./configs/test/mnist_backdoor_dp_configs.json


sed -i 's/"name": "GaussianDP"/"name": "LaplaceDP"/g' ./configs/test/mnist_backdoor_dp_configs.json
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/test/mnist_backdoor_dp_configs.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/test/mnist_backdoor_dp_configs.json

# srun --gres=gpu:a100-80G:1 --time=1-00:00:00 --mem=20000 bash ./local_test/mnist_backdoor_dp.sh
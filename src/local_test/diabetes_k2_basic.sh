#!/bin/bash

# ###### make sure starts with non-trainable head ######
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/diabetes_basic_k2.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/diabetes_basic_k2.json

echo 'diabetes_k2_basic begin'
# non-trainable
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 2/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 2/"iteration_per_aggregation": 3/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 3/"iteration_per_aggregation": 4/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 4/"iteration_per_aggregation": 5/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 6/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 6/"iteration_per_aggregation": 7/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 7/"iteration_per_aggregation": 8/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 8/"iteration_per_aggregation": 9/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 9/"iteration_per_aggregation": 10/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 10/"iteration_per_aggregation": 1/g' ./configs/diabetes_basic_k2.json


# ###### change to trainable head ######
sed -i 's/"apply_trainable_layer": 0/"apply_trainable_layer": 1/g' ./configs/diabetes_basic_k2.json
sed -i 's/"global_model": "ClassificationModelHostHead"/"global_model": "ClassificationModelHostTrainableHead"/g' ./configs/diabetes_basic_k2.json


for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 2/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 2/"iteration_per_aggregation": 3/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 3/"iteration_per_aggregation": 4/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 4/"iteration_per_aggregation": 5/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 6/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 6/"iteration_per_aggregation": 7/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 7/"iteration_per_aggregation": 8/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 8/"iteration_per_aggregation": 9/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 9/"iteration_per_aggregation": 10/g' ./configs/diabetes_basic_k2.json
for i in `seq 97 106`; do 
    python main_separate.py --configs diabetes_basic_k2 --seed $i
done

sed -i 's/"iteration_per_aggregation": 10/"iteration_per_aggregation": 1/g' ./configs/diabetes_basic_k2.json

# ###### change back to non-trainable head ######
sed -i 's/"apply_trainable_layer": 1/"apply_trainable_layer": 0/g' ./configs/diabetes_basic_k2.json
sed -i 's/"global_model": "ClassificationModelHostTrainableHead"/"global_model": "ClassificationModelHostHead"/g' ./configs/diabetes_basic_k2.json
echo 'diabetes_k2_basic done'
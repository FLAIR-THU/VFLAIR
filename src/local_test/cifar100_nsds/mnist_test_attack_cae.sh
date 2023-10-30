echo  'CAE/DCAE nsds Begin' #SBATCH --qos high
#CAE 1.0
python cifar100_nsds.py --configs cifar100_nsds/cae

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/cifar100_nsds/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|g' ./configs/cifar100_nsds/cae.json
python cifar100_nsds.py --configs cifar100_nsds/cae

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/cifar100_nsds/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|g' ./configs/cifar100_nsds/cae.json
python cifar100_nsds.py --configs cifar100_nsds/cae

# CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/cifar100_nsds/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|g' ./configs/cifar100_nsds/cae.json
python cifar100_nsds.py --configs cifar100_nsds/cae


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/cifar100_nsds/cae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|"model_path": ""../src/evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704""|g' ./configs/cifar100_nsds/cae.json

echo  'CAE End'

echo  'DCAE agg nsds Begin' #SBATCH --qos high
#CAE 1.0
python cifar100_nsds.py --configs cifar100_nsds/dcae

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/cifar100_nsds/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|g' ./configs/cifar100_nsds/dcae.json
python cifar100_nsds.py --configs cifar100_nsds/dcae

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/cifar100_nsds/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.5_1636175420"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|g' ./configs/cifar100_nsds/dcae.json
python cifar100_nsds.py --configs cifar100_nsds/dcae

# CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/cifar100_nsds/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.1_1636175237"|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|g' ./configs/cifar100_nsds/dcae.json
python cifar100_nsds.py --configs cifar100_nsds/dcae


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/cifar100_nsds/dcae.json
sed -i 's|"model_path": "../src/evaluates/defenses/trained_CAE_models/autoencoder_2_0.0_1636174878"|"model_path": ""../src/evaluates/defenses/trained_CAE_models/autoencoder_2_1.0_1636175704""|g' ./configs/cifar100_nsds/dcae.json

echo  'CAE End'
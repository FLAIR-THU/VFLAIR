echo  'DCAE main renew Begin'
#CAE 1.0
python main_pipeline.py --configs test_zty4_2dcae

# # CAE 0.5
sed -i 's/"lambda": 1.0/"lambda": 0.5/g' ./configs/test_zty4_2dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_1.0_1642396548"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.5_1642396797"|g' ./configs/test_zty4_2dcae.json
python main_pipeline.py --configs test_zty4_2dcae

# # CAE 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/test_zty4_2dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.5_1642396797"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.1_1686717045"|g' ./configs/test_zty4_2dcae.json
python main_pipeline.py --configs test_zty4_2dcae

# CAE 0.0
sed -i 's/"lambda": 0.1/"lambda": 0.0/g' ./configs/test_zty4_2dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.1_1686717045"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.0_1631093149"|g' ./configs/test_zty4_2dcae.json
python main_pipeline.py --configs test_zty4_2dcae


sed -i 's/"lambda": 0.0/"lambda": 1.0/g' ./configs/test_zty4_2dcae.json
sed -i 's|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_0.0_1631093149"|"model_path": "./evaluates/defenses/trained_CAE_models/autoencoder_10_1.0_1642396548"|g' ./configs/test_zty4_2dcae.json

echo  'DCAE End'
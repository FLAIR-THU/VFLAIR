echo "MNIST"
python3 main_pipeline.py --configs main_task/2_party/aggVFL/paillier_configs_mnist
python3 main_paillier.py --configs main_task/2_party/aggVFL/paillier_configs_mnist --debug

echo "Nursery - LR"
python3 main_pipeline.py --configs main_task/2_party/aggVFL/paillier_configs_nursery_lr
python3 main_paillier.py --configs main_task/2_party/aggVFL/paillier_configs_nursery_lr --debug

echo "Nursery - NN"
python3 main_pipeline.py --configs main_task/2_party/aggVFL/paillier_configs_nursery_nn_agg
python3 main_paillier.py --configs main_task/2_party/aggVFL/paillier_configs_nursery_nn_agg --debug

echo "Credit - LR"
python3 main_pipeline.py --configs main_task/2_party/aggVFL/paillier_configs_credit_lr
python3 main_paillier.py --configs main_task/2_party/aggVFL/paillier_configs_credit_lr --debug

echo "Credit - NN"
python3 main_pipeline.py --configs main_task/2_party/aggVFL/paillier_configs_credit_nn
python3 main_paillier.py --configs main_task/2_party/aggVFL/paillier_configs_credit_nn --debug

echo 'mnist quant'

python main_pipeline_vanilla.py --configs vanilla/mnist

sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 5/g' ./configs/vanilla/mnist.json
sed -i 's/"lr": 0.01/"lr": 0.005/g' ./configs/vanilla/mnist.json

python main_pipeline_vanilla.py --configs vanilla/mnist

sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 1/g' ./configs/vanilla/mnist.json
sed -i 's/"lr": 0.005/"lr": 0.01/g' ./configs/vanilla/mnist.json
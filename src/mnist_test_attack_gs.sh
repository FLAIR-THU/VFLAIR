echo 'GS main latest Begin'
# 100.0 #SBATCH --qos high
# python main_pipeline.py --configs test_zty4_2gs

# 99.5
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/test_zty4_2gs.json
python main_pipeline.py --configs test_zty4_2gs --gpu 6

# 99
sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99.0/g' ./configs/test_zty4_2gs.json
python main_pipeline.py --configs test_zty4_2gs --gpu 6

sed -i 's/"gradient_sparse_rate": 99.0/"gradient_sparse_rate": 98.0/g' ./configs/test_zty4_2gs.json

# 97
sed -i 's/"gradient_sparse_rate": 98.0/"gradient_sparse_rate": 97.0/g' ./configs/test_zty4_2gs.json
python main_pipeline.py --configs test_zty4_2gs --gpu 6

sed -i 's/"gradient_sparse_rate": 97.0/"gradient_sparse_rate": 96.0/g' ./configs/test_zty4_2gs.json

# 95
sed -i 's/"gradient_sparse_rate": 96.0/"gradient_sparse_rate": 95.0/g' ./configs/test_zty4_2gs.json
python main_pipeline.py --configs test_zty4_2gs --gpu 6

sed -i 's/"gradient_sparse_rate": 95.0/"gradient_sparse_rate": 100.0/g' ./configs/test_zty4_2gs.json
echo 'GS End'


# Begin with LaplaceDP 0.0001 
echo 'LDP agg nsds Begin'
# DP 0.0001
python main_pipeline.py --configs test_zty4_2dp --gpu 6

# DP 0.001
sed -i 's/"dp_strength": 0.0001/"dp_strength": 0.001/g' ./configs/test_zty4_2dp.json
python main_pipeline.py --configs test_zty4_2dp --gpu 6

# DP 0.01
sed -i 's/"dp_strength": 0.001/"dp_strength": 0.01/g' ./configs/test_zty4_2dp.json
python main_pipeline.py --configs test_zty4_2dp --gpu 6

# DP 0.1
sed -i 's/"dp_strength": 0.01/"dp_strength": 0.1/g' ./configs/test_zty4_2dp.json
python main_pipeline.py --configs test_zty4_2dp --gpu 6


sed -i 's/"dp_strength": 0.1/"dp_strength": 0.0001/g' ./configs/test_zty4_2dp.json
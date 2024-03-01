#!/bin/bash
#SBATCH --account=def-khill22
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000M
#SBATCH --time=01:00:00

module avail python
module load python/3.10
#virtualenv --no-download ENV
source ENV/bin/activate

#pip install -r requirements.txt
python3 src/cluster.py --dataset="1_Cypriniformes" --pairs_file_name="train_data_frag_0.95_0.8.pkl" --lr=0.0002 --temp_ins=0.1 --temp_clu=1.0 --number_of_models=10
python3 src/cluster.py --dataset="2_Cyprinoidei" --pairs_file_name="train_data_frag_0.95_0.8.pkl" --lr=0.0002 --temp_ins=0.1 --temp_clu=1.0 --number_of_models=10
python3 src/cluster.py --dataset="3_Cyprinidae" --pairs_file_name="train_data_frag_0.95_0.8.pkl" --lr=0.0002 --temp_ins=0.1 --temp_clu=1.0 --number_of_models=10
python3 src/cluster.py --dataset="4_Cyprininae" --pairs_file_name="train_data_frag_0.95_0.8.pkl" --lr=0.0002 --temp_ins=0.1 --temp_clu=1.0 --number_of_models=10
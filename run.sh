#!/bin/bash
#SBATCH --account=def-khill22
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000M
#SBATCH --time=05:00:00

module avail python
module load python/3.10
#virtualenv --no-download ENV
source ENV/bin/activate

#pip install -r requirements.txt
python3 src/cluster.py --dataset="2_Cyprinoidei" --lr=7e-5 --pairs_file_name="train_data_mutate_e-4_e-3.pkl"
python3 src/cluster.py --dataset="2_Cyprinoidei" --lr=7e-5 --pairs_file_name="train_data_frag_0.95_0.8.pkl"
python3 src/cluster.py --dataset="3_Cyprinidae" --lr=7e-5 --pairs_file_name="train_data_mutate_e-4_e-3.pkl"
python3 src/cluster.py --dataset="3_Cyprinidae" --lr=7e-5 --pairs_file_name="train_data_frag_0.95_0.8.pkl"
python3 src/cluster.py --dataset="4_Cyprininae" --lr=7e-5 --pairs_file_name="train_data_mutate_e-4_e-3.pkl"
python3 src/cluster.py --dataset="4_Cyprininae" --lr=7e-5 --pairs_file_name="train_data_frag_0.95_0.8.pkl"


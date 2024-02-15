#!/bin/bash
#SBATCH --account=def-khill22
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000M
#SBATCH --time=01:00:00

module avail python
module load python/3.10
virtualenv --no-download ENV
source ENV/bin/activate

pip install -r requirements.txt
python3 src/cluster.py
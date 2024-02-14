#!/bin/bash
#SBATCH --account=def-khill-22
#SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000M
#SBATCH --time=01:00:00

module avail python
module load python/3.10
virtualenv --no-download ENV
source ENV/bin/activate

pip install -r requirement.txt
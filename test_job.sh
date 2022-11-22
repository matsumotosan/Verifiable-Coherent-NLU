#!/bin/bash

#SBATCH --account=eecs595f22_class
#SBATCH --partition=gpu
#SBATCH --time=00-03:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=90G
#SBATCH --account=eecs595f22_class
#SBATCH --mail-type=BEGIN,END,FAIL

# set up job
module load python/3.9.12 cuda
pushd /home/matsumos/Verifiable-Coherent-NLU
source venv/bin/activate

# run job
python3 main.py

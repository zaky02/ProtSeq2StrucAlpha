#!/bin/bash

#SBATCH --job-name=protseq2struc
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/protseq/%j.out
#SBATCH --error=logs/protseq/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --qos=acc_debug

module purge
ml bsc/1.0
ml intel/2023.2.0
ml cmake/3.25.1
ml impi/2021.10.0
ml mkl/2023.2.0
ml miniconda/24.1.2
ml anaconda

eval "$(conda shell.bash hook)"
source activate ProtSeq2StrucAlpha

wandb login c707377256b2e57dbb0b42bd3c36744b3d5617c8 
wandb offline

srun python -u train.py --config config.json --dformat csv

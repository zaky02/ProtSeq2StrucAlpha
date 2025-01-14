#!/bin/bash

#SBATCH --account=bsc72
#SBATCH --qos=acc_bscls     # Quality of Service or gp_bscls (gp is for cpu) acc_debug acc_bscls (acc is for gpu)
#SBATCH --job-name=test      # Job name (change it)
#SBATCH --output=logs/test/%j.out     # Standard output and error log (change it)
#SBATCH --error=logs/test/%j.err        # Standard error file output (change it)
#SBATCH --cpus-per-task=20          # Number of CPU cores per task (maximum cpus is 80, must multiply by the number of tasks)
#SBATCH --time=20:00:00            # Time limit hrs:min:sec
#SBATCH --gres=gpu:4                # Request 1 GPU
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks-per-node=4                  # Request 1 task (number of tasks must be as many as GPUs)

ml impi mkl intel
source activate ProtSeq2StrucAlpha

# Log in to WandB
echo "WandB"
wandb login    # write it in the terminal to start wandb

export WANDB_API_KEY=c707377256b2e57dbb0b42bd3c36744b3d5617c8
export WANDB_RUN_GROUP=$SLURM_JOB_ID
export WANDB_CONSOLE=off
export WANDB_DEBUG=TRUE

export CUDA_VISIBLE_DEVICES=0,1,2,3
NCCL_DEBUG=INFO
export TIMELIMIT=20:00:00    # maximum is 48h but if we reduce the amount of hours it goes in front of other jobs 
echo $QOS

set +e
srun ~/.conda/envs/ProtSeq2StrucAlpha/bin/python train.py --config config.json --dformat csv
set -e
echo "Finished"

# sbatch run.sh (this is to execute the file)

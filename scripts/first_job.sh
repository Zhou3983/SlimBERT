#!/bin/bash
#SBATCH --account=def-karray
#SBATCH --gres=gpu:p100:1              # Number of GPUs (per node)
#SBATCH --mem=12G               # memory (per node)
#SBATCH --time=0-03:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=2

export SLURM_ACCOUNT=def-karray
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT

python /home/rickxu/projects/def-karray/rickxu/BottledBERT/main.py

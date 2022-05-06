#!/bin/bash -l
# Send an email
#$ -m ea
# Set Project
#$ -P dl523
# Request CPUs
#$ -pe omp 4
# Request GPUs
#$ -l gpus=1
# Minimum GPU requirement
#$ -l gpu_c=7

module load python3/3.8.10
module load pytorch/1.9.0
python train.py
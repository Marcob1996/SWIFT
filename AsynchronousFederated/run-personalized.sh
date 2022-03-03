#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=AsyncDecentralized     # sets the job name if not set from environment
#SBATCH --time=00:05:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --mem 0gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.2.2

mpirun -np 4 python Train.py  --graph fully-connected --num_clusters 1 --personalize 1 --max_sgd 10 --resSize 50 --bs 64 --epoch 15 --name comet_test --description asyncDecentralized --randomSeed 9001 --datasetRoot ./data --outputFolder OutputPersonalized

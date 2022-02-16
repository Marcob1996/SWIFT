#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=AsyncDecentralized     # sets the job name if not set from environment
#SBATCH --time=00:45:00                   # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger               # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                   # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --mem 32gb                        # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END                   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi

mpirun -np 2 python Train.py --description InitialAsyncDecentralized --randomSeed 9001 --datasetRoot ./data  --outputFolder Output --downloadCifar 1 --epoch 10 --name initial_run --resSize 50

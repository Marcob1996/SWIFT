#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=PDSGD     # sets the job name if not set from environment
#SBATCH --time=05:30:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 5 python Train.py  --graph fully-connected --name ldsgd-0.5-noniid-fc-4-test1 --comm_style ld-sgd --i1 1 --i2 2 --degree_noniid 0.5 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name ldsgd-0.5-noniid-fc-4-test2 --comm_style ld-sgd --i1 1 --i2 2 --degree_noniid 0.5 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name ldsgd-0.5-noniid-fc-4-test3 --comm_style ld-sgd --i1 1 --i2 2 --degree_noniid 0.5 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name ldsgd-0.5-noniid-fc-4-test4 --comm_style ld-sgd --i1 1 --i2 2 --degree_noniid 0.5 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name ldsgd-0.5-noniid-fc-4-test5 --comm_style ld-sgd --i1 1 --i2 2 --degree_noniid 0.5 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output

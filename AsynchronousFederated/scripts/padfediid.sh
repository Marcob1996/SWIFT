#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=PadFed     # sets the job name if not set from environment
#SBATCH --time=10:45:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=high    # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

# module load openmpi
# module load cuda/11.1.1

mpirun -np 16 python Train.py --name padfed-wb-iid1 --graph ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name padfed-wb-iid2 --graph ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name padfed-wb-iid3 --graph ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name padfed-wb-iid4 --graph ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name padfed-wb-iid5 --graph ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output

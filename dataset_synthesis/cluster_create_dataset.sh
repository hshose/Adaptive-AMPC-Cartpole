#!/bin/bash

### Job name
#SBATCH --job-name=pendulum_samplempc
#SBATCH --account=rwth1570

#SBATCH --array=1-800%800

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH --time=02:30:00

### CPUS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

### File for the output
#SBATCH --output=/home/hh753317/projects/dsme/pendulum-pyomo/logs/Cluster.%J.log

### The last part consists of regular shell commands:
source /home/hh753317/.bashrc

cd /home/hh753317/projects/dsme/pendulum-pyomo

python3 main_sample.py parallel_sample \
    --node_number=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --instances=12 \
    --samplesperinstance=50 

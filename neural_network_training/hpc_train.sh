#!/usr/local_rwth/bin/zsh
#
#SBATCH --job-name=train_dnni
#SBATCH --output=/home/mf724021/hpc_data/slurm_output/%A_%a.out
#SBATCH --nodes=1 # request one node
#SBATCH --account=rwth1483
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexander.graefe@dsme.rwth-aachen.de
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3GB # memory per node
#SBATCH --time=3-00:00:00
#SBATCH --array=0-10

module load Python/3.10.4

source .venv/bin/activate
python train_hpc.py -i $SLURM_ARRAY_TASK_ID
deactivate
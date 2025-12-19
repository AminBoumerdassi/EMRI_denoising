#!/bin/bash -e   
#SBATCH --job-name=EMRI_training   # job name (shows up in the queue)
#SBATCH --time=00-03:30:00  # Walltime (DD-HH:MM:SS)
#SBATCH --partition=skylake
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1   # number of CPUs per task (1 by default)
#SBATCH --mem=8G         # amount of memory per node (1 by default)
#SBATCH --output=/fred/oz303/aboumerd/EMRI_denoising/output_logs/slurm-%j.out

cd /fred/oz303/aboumerd/EMRI_denoising

# load required modules and environments
source /fred/oz303/aboumerd/EMRI_denoising/init_env.sh

# run the training script
python -u /fred/oz303/aboumerd/EMRI_denoising/train_CNN_on_EMRIs.py
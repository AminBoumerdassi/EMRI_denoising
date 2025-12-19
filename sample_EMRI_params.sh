#!/bin/sh

#SBATCH --job-name=EMRI_param_sampling
#SBATCH --output=output_logs/output_%j.out
#SBATCH --error=error_logs/error_%j.err

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=skylake

#SBATCH --gres=gpu

#SBATCH --mem=8G
#SBATCH --time=00:60:00

cd /fred/oz303/aboumerd/EMRI_denoising
source /fred/oz303/aboumerd/EMRI_denoising/init_env.sh
python -u sample_EMRI_params.py
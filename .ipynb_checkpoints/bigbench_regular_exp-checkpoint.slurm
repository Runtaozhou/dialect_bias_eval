#!/bin/bash
#SBATCH -A hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
##SBATCH --constraint=a100_80gb

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-30:00:00
#SBATCH -J llm_bias
#SBATCH -o /home/uar6nw/Documents/slurm_outputs/llm_output-%j.out
#SBATCH -e /home/uar6nw/Documents/slurm_outputs/llm_error-%j.err

module purge
module load anaconda
source activate base
export PYTHONPATH="/home/uar6nw/Documents/LLM_Persona:$PYTHONPATH"

ollama serve &

python bigbench_regular_exp.py
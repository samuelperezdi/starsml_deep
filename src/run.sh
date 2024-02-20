#!/bin/bash
#
#SBATCH --job-name=trainingstars
#SBATCH -p gpu
##SBATCH -N 1
##SBATCH -n 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 20-00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@urosario.edu.co

#module load miniconda3/4.8.3
#source activate /datacnmat01/macc/usuarios/victor.perez/.conda/envs/cross-matching

srun python3 train.py -exp run0 -dev gpu

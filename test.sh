#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1

python test.py
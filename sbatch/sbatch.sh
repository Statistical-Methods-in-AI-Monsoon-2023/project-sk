#!/bin/bash
#SBATCH -A neuro 
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=###OUTPUT###
#SBATCH --partition=ihub
#SBATCH --mail-user=soham.korade@students.iiit.ac.in
#SBATCH --mail-type=ALL
module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2

###HERE###

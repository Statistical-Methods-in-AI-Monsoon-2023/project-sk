#!/bin/bash
#SBATCH -A neuro 
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=darknet_file.txt
module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2

python3 
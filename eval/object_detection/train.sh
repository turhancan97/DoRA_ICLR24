#!/bin/bash
#SBATCH -p all # partition (queue).
#SBATCH -N 1 # number of nodes
#SBATCH -n 8
#SBATCH --gres=gpu:01
#SBATCH -w xeon-e-v100-02
#SBATCH --job-name=vit
#SBATCH --output=logs/results_train_vit.txt
#SBATCH --error=logs/errors_vit.txt

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dora

python -m torch.distributed.launch eval_object_detect.py 
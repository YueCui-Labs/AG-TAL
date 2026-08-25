#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

dataset_ID=$1
fold_num=$2
tr_model=$3


nnUNetv2_train $dataset_ID 3d_fullres $fold_num -tr $tr_model -num_gpus 1 -device cuda

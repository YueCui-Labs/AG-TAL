#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

network=$1
input_dir=$2
output_dir=$3
fold_num=$4

nnUNetv2_predict -i $input_dir -o $output_dir -c 3d_fullres -f 0 -tr $network -d $fold_num -chk checkpoint_best.pth -device cuda 




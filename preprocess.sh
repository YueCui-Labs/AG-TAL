#!/bin/bash
#SBATCH --job-name=nnU_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


dataset_num=$1
nnUNetv2_plan_and_preprocess -d $dataset_num --verify_dataset_integrity

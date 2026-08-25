#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --cpus-per-task=1

subject=$1
echo $subject
data_dir=$2
output_dir=$3
# Loop through subjects and submit a job for each one

for sub in `ls $data_dir/$subject`;do
echo $sub

input_path=$data_dir/$subject/$sub/TOF_${subject}_${sub}.nii.gz
label_path=$data_dir/$subject/$sub/TOF_${subject}_${sub}_label.nii.gz


output_prefix=${subject}_${sub}
reg_dir=$output_dir/${subject}_${sub}

moving_image=./Template/AVG_TOF_MNI_down.nii.gz
cube_image=./Template/cube_template_size.nii.gz

mkdir -p $reg_dir/antsReg/
cube_path=$reg_dir/antsReg/willis_cube_in_individual.nii.gz
if [ -e "$cube_path" ];
then
    echo $output_prefix 'registeration done.'
else

python -u ./reorientation_RAS.py  $input_path  $reg_dir/TOF_RAS.nii.gz
python -u ./reorientation_RAS.py  $label_path  $reg_dir/TOF_RAS_label.nii.gz


antsRegistrationSyNQuick.sh -d 3 -f $reg_dir/TOF_RAS.nii.gz -m $moving_image -o $reg_dir/antsReg//WILLIS_ANTS -t a -n 20

antsApplyTransforms -i $cube_image -r $reg_dir/TOF_RAS.nii.gz -o $reg_dir/antsReg/willis_cube_in_individual.nii.gz -t $reg_dir/antsReg/WILLIS_ANTS0GenericAffine.mat

fi
output_tof=$output_dir/imagesTr/${output_prefix}_0000.nii.gz
output_label=$output_dir/labelsTr/${output_prefix}.nii.gz

mkdir -p $output_dir/imagesTr
mkdir -p $output_dir/labelsTr
python -u ./crop_cube.py $reg_dir/TOF_RAS.nii.gz $reg_dir/TOF_RAS_label.nii.gz $reg_dir/antsReg/willis_cube_in_individual.nii.gz $output_tof $output_label $reg_dir/min_coords.txt

done

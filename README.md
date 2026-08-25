# AG-TAL

## ROI cropping

Bash file: ./ROI_extraction.sh

Function: 1. Rotate to RAS direction; 2. Perform linear registration from the template to the individual space; 3. Obtain the registered ROI cube region, cut this region from the original image and save the ROI

Input parameters: 1. Dataset name; 2. The path where all datasets are stored; 3. The path for saving files

Requirements: The data folder within the dataset folder has the format: dataset_dir/subject_name/TOF_{dataset_name}_{subject_name}.nii.gz, dataset_dir/subject_name/TOF_{dataset_name}_{subject_name}_label.nii.gz

## Data preprocessing

execution bash file: ./preprocess.sh

Function: Calls the nnUNetv2_plan_and_preprocess command, automatically analyzes the images within the dataset, completes preprocessing, and the preprocessing results will be saved in the Dataset[XXX]_[dataset_name] folder within nnUNet_preprocessed

Input parameters: 1. Dataset number [XXX]

## Dataset training

Data training bash file: ./preprocess.sh

Input parameters: 1. Dataset number [XXX]; 2. Training data fold number; 3. Training model name (default: nnUNetTrainer, AG-TAL network: nnUNetTrainer_ResEncoder_AGTAL_AdjV3_NeighclDice_NoMirroring)

## Dataset inference

Data inference file: ./predict.sh

Input parameters: 1. Training model name (default: nnUNetTrainer); 2. Folder containing the ROI data to be inferred; 3. Folder for saving segmentation output; 4. Dataset number [XXX];

Segmentation result images and calculated segmentation metrics are all saved in the segmentation output folder

import nibabel as nib
import numpy as np
import pandas as pd
import os
import argparse

def ReOrientation_to_RAS(input_path, output_path):
    print('input path:',input_path)
    tof_raw: nib.Nifti1Image = nib.load(input_path)
    print(f"Original orientation: {nib.aff2axcodes(tof_raw.affine)}")

    tof = nib.as_closest_canonical(tof_raw)
    print(f"Transformed orientation: {nib.aff2axcodes(tof.affine)}")
    tof.to_filename(output_path)
  
  
  
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "brain_path",
        help="input_path.",
        type=str,
    )
    parser.add_argument(
        "output_path",
        help="output_prefix.",
        type=str,
    )
    args = parser.parse_args()
    brain_path=args.brain_path
    output_path=args.output_path
    ReOrientation_to_RAS(input_path=brain_path,
                         output_path=output_path)
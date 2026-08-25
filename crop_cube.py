import nibabel as nib
import numpy as np
import pandas as pd
import os
import argparse
import SimpleITK as sitk
from nipype.interfaces import afni as afni


def AutoBox(input_file, output_file):

    abox = afni.Autobox()
    abox.inputs.in_file = input_file
    abox.inputs.out_file = output_file
    abox.inputs.padding = 6
    abox.inputs.args = "-overwrite"
    abox.inputs.no_clustering=True
    res = abox.run()
    return res.outputs.get()

def crop_cube_label(cube_file,brain_file,output_file):
        # """
        # Crop the cube for training.
        # if cube can not cover all labeld voxels, will dilate the cube until cover.

        # Args:
        #     cube_file (str): the COW cube file.
        #     brain_file (str): the brain file for cropping.
        #     output_file (str): Where to save the train cube result.

        # Returns:
        #     np.ndarray: the upper left coordinate[sag, coro, axial].
        # """

    input_image = sitk.ReadImage(brain_file)
    input_data = sitk.GetArrayFromImage(input_image)

    cube_image = sitk.ReadImage(cube_file)
    cube = sitk.GetArrayFromImage(cube_image)
    original_spacing = cube_image.GetSpacing()
    original_origin = cube_image.GetOrigin()
    original_direction = cube_image.GetDirection()

    # Finding bounding index of cube

    (sag, coro, axial) = np.nonzero(cube)
    sag_min = sag.min()
    sag_max = sag.max()
    coro_min = coro.min()
    coro_max = coro.max()
    axial_min = axial.min()
    axial_max = axial.max()

    res=AutoBox(input_file=brain_file, output_file=output_file)
    x_min=res['x_min']
    y_min=res['y_min']
    z_min=res['z_min']

    x_max=res['x_max']
    y_max=res['y_max']
    z_max=res['z_max']

    # x_size=x_max-x_min
    # y_size=y_max-y_min
    # z_size=z_max-z_min
    print('z_min:',z_min, ' z_max:',z_max)
    print('sag_min:',sag_min, ' sag_max:',sag_max)
    if z_min<0:
        z_min=0

    if z_min-sag_min>50:
        sag_min=z_min
        print('crop.')
    else:
        sag_min=min(sag_min,z_min)
    
    coro_min=min(coro_min,y_min)
    axial_min=min(axial_min,x_min)

    sag_max=max(sag_max,z_max)
    coro_max=max(coro_max,y_max)
    axial_max=max(axial_max,x_max)


    

    cube_tof = input_data[sag_min:sag_max, coro_min:coro_max, axial_min:axial_max]
    
    cube_shape=cube_tof.shape
    print(cube_shape)

    # out_size = [int(sag_max - sag_min), int(coro_max - coro_min), int(axial_max - axial_min)]

    cube_new_image = sitk.GetImageFromArray(cube_tof)
    cube_new_image.SetSpacing(original_spacing)
    # cube_new_image.SetSize(out_size)
    cube_new_image.SetDirection(original_direction)
    cube_new_image.SetOrigin(original_origin)

    sitk.WriteImage(cube_new_image, output_file)

    return np.array([axial_min,coro_min,sag_min]),cube_shape

def crop_cube(brain_file,output_file,min_coord,cube_shape):


    input_image = sitk.ReadImage(brain_file)
    input_data = sitk.GetArrayFromImage(input_image)

    
    original_spacing = input_image.GetSpacing()
    original_origin = input_image.GetOrigin()
    original_direction = input_image.GetDirection()

    # Finding bounding index of cube
    axial_min=min_coord[0]
    coro_min=min_coord[1]
    sag_min=min_coord[2]
    
    sag_size=cube_shape[0]
    coro_size=cube_shape[1]
    axial_size=cube_shape[2]

    cube_tof = input_data[sag_min:sag_size+sag_min, coro_min:coro_size+coro_min, axial_min:axial_size+axial_min]
    
    cube_new_image = sitk.GetImageFromArray(cube_tof)
    cube_new_image.SetSpacing(original_spacing)
    # cube_new_image.SetSize(out_size)
    cube_new_image.SetDirection(original_direction)
    cube_new_image.SetOrigin(original_origin)

    sitk.WriteImage(cube_new_image, output_file)

    return np.array([axial_min,coro_min,sag_min])
    
  
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "brain_path",
        help="input_path.",
        type=str,
    )
    parser.add_argument(
        "label_path",
        help="input_path.",
        type=str,
    )
    parser.add_argument(
        "cube_path",
        help="input_path.",
        type=str,
    )
    parser.add_argument(
        "output_tof",
        help="output_prefix.",
        type=str,
    )
    parser.add_argument(
        "output_label",
        help="output_prefix.",
        type=str,
    )
    parser.add_argument(
        "min_coords_txt",
        help="output_prefix.",
        type=str,
    )
    args = parser.parse_args()
    brain_path=args.brain_path
    label_path=args.label_path
    output_tof_file=args.output_tof
    output_label_file=args.output_label
    cube_path=args.cube_path
    min_coords_txt=args.min_coords_txt
    min_coords,cube_shape=crop_cube_label(brain_file=label_path,
                                          cube_file=cube_path,
                                          output_file=output_label_file)
    min_coords=crop_cube(brain_file=brain_path,
                         output_file=output_tof_file,
                         min_coord=min_coords,
                         cube_shape=cube_shape)
    np.savetxt(min_coords_txt,min_coords,delimiter=',')
    
import pandas as pd
import numpy as np
import os
import argparse
import nibabel as nib
import copy

# import SimpleITK as sitk
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.morphology import binary_dilation
from skimage.morphology import diamond
import torch
import torch.nn as nn
from typing import Callable
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
import traceback


def Tube_skeleton(gt):
    # print('gt.shape:', gt.shape)
    # print('gt.unique:',np.unique(gt))
    gt_bin=np.zeros(gt.shape, dtype=np.bool_)
    gt_bin[gt>0]=1
    gt_skel=skeletonize_3d(gt_bin)
    # print('gt_skel.shape:',gt_skel.shape)
    kernel_= diamond(radius=2, dtype=np.bool_)
    kernel_3d=np.zeros((5,5,5), dtype=np.bool_)
    for i in range(5):
        kernel_3d[i]=kernel_
    # print(kernel_3d)
    tube_skel= binary_dilation(gt_skel, kernel_3d)
    if len(np.unique(gt))>2: # multi class
        gt_tube_skel = tube_skel * gt
    else:
        gt_tube_skel = tube_skel
    return gt_tube_skel

class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftSkeletonRecallLoss, self).__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                
                y_onehot = y_onehot[:, 1:]
    
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        rec = rec.mean()
        return -rec
















if __name__ == '__main__':

    vessel_path=os.path.join('W:/data/MIDAS-process/CW_seg/Label_data/MICCAI_186', 'label_merge_20_edit.nii.gz')
    vessel_=nib.load(vessel_path)
    gt_image=vessel_.get_fdata()

    gt_tube=Tube_skeleton(gt_image)
    save_path=os.path.join('W:/data/MIDAS-process/CW_seg/Label_data/MICCAI_186', 'label_merge_20_edit_TubeSkel.nii.gz')
    nib.save(nib.Nifti1Image(gt_tube, vessel_.affine, vessel_.header), save_path)
    print('done.')

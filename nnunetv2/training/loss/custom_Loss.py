from multiprocessing import reduction
from turtle import forward
from nnunetv2.training import loss
import pandas as pd
import numpy as np
import os
import argparse
import nibabel as nib
import copy

from skimage.morphology import skeletonize, skeletonize_3d
from skimage.morphology import binary_dilation
from skimage.morphology import diamond
from sympy import li
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
import traceback

import torch



def gaussian_kernel(size, sigma):

    x = np.arange(0, size[0], 1, float)
    y = np.arange(0, size[1], 1, float)
    z = np.arange(0, size[2], 1, float)
    x, y, z = np.meshgrid(x, y, z)

    d = np.sqrt(x**2 + y**2 + z**2)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))

    k=g / g.sum()
    # kernel = gaussian_kernel(size=(7, 7, 7), sigma=3.0)
    # kernel = torch.tensor(kernel, dtype=keypoint_tensor.dtype, device=keypoint_tensor.device)
    kernel = torch.FloatTensor(k).unsqueeze(0)
    kernel=kernel.repeat(12,1,1,1,1)
    return kernel

def apply_gaussian_kernel(keypoint_tensor, kernel, kernel_size=(3, 3, 3), sigma=1.0):

  
    blurred_keypoint = nn.functional.conv3d(keypoint_tensor.float(), kernel, padding=kernel_size[0]//2, groups=keypoint_tensor.shape[1])
    print('blurred_keypoint.shape:',blurred_keypoint.shape)
    
    return blurred_keypoint






class SoftKeyPointRegressionlLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftKeyPointRegressionlLoss, self).__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        
        self.kernel=gaussian_kernel(size=(7,7,7), sigma=3.0)
        # self.loss_=nn.MSELoss()
        

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape
        print('shp_x:',shp_x)

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
                print('gt.unique:',torch.unique(gt))
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]
                self.kernel=self.kernel.to(y_onehot.device)
                y_onehot=apply_gaussian_kernel(y_onehot,self.kernel, (7, 7, 7), 3.0)


        # loss_=self.loss_(x,y_onehot)
        print('x.shape:',x.shape)
        print('y_onehot.shape:',y_onehot.shape)

        loss_every=(x-y_onehot)**2
        mse_per_channel=loss_every.mean(dim=[0,2,3,4])

        loss_=mse_per_channel.mean()

        return loss_


# radius-aware dice loss
class MemoryEfficientSoftDiceLoss_Weight(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss_Weight, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, weight, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape # radius weight map

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = (y_onehot*weight).sum(axes) if loss_mask is None else (y_onehot*weight* loss_mask).sum(axes)
 

        
        # add weight map into the dice loss
        intersect = (x * y_onehot *weight).sum(axes) if loss_mask is None else (x * y_onehot * weight* loss_mask).sum(axes)
        sum_pred = (x*weight).sum(axes) if loss_mask is None else (x *weight* loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            intersect = AllGatherGrad.apply(intersect).sum(0)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        try:

            dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        except Exception as error:
            print(error)
            print('\n','>>>' * 20)
            print(traceback.print_exc())


        dc=-dc.mean()

        return dc     


        
# adjacency-aware dice loss (MICCAI version)
class GraphMatchVoxelLoss_Weight(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True,use_add=False):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(GraphMatchVoxelLoss_Weight, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.use_add=use_add

        self.connection_list=[[2,3,15,16],
                [1,19,8],
                [1,20,9],
                [5,8,11,13],
                [4],
                [7,9,12,14],
                [6],
                [2,4,19],
                [3,6,20],
                [11,12,17,18],
                [4,10,17],
                [6,10,18],
                [4],
                [6],
                [1],
                [1],
                [10,11],
                [10,12],
                [2,8],
                [3,9]]


    
    def soft_dilate(self,img):
        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


    def get_fp_fn(self,x,y,num_classes):
        shp=x.shape

        fp_mask_all=torch.zeros_like(x,dtype=torch.int32,device=x.device)
        fn_mask_all=torch.zeros_like(x,dtype=torch.int32,device=x.device)
        
        for b in range(shp[0]):
            vessel_=x[b]

            y_vessel=y[b]

            labels_x=torch.unique(vessel_.int())
            labels_y=torch.unique(y_vessel.int())

            fp_mask=torch.zeros_like(vessel_,dtype=torch.int32,device=x.device)
            fn_mask=torch.zeros_like(vessel_,dtype=torch.int32,device=x.device)
            

            for lab in range(num_classes):
                if lab==0:
                    continue

                if (lab not in labels_y) and (lab not in labels_x):
                    continue
                elif lab not in labels_y:

                    fp_mask[torch.where(vessel_==lab)]=1
                elif lab not in labels_x:

                    fn_mask[torch.where(y_vessel==lab)]=1
                else:

                    lab_vessel=torch.zeros_like(vessel_,dtype=torch.float32,device=x.device)
                    lab_vessel[torch.where(vessel_==lab)]=1
                    lab_vessel=lab_vessel.unsqueeze(0)

                    dilate_lab_vessel=self.soft_dilate(lab_vessel)
                    dilate_lab_vessel=dilate_lab_vessel.int()
                    dilate_lab_vessel=dilate_lab_vessel.squeeze(0)

                    neigh_vessel=dilate_lab_vessel*vessel_ # 找pred当前类别的邻居
                    neigh_list=torch.unique(neigh_vessel)

                    neigh_y_list=torch.tensor(self.connection_list[lab-1],device=x.device)
                    x_not_in_y=torch.isin(neigh_list,neigh_y_list,invert=True) #被pred预测错误的邻居
                    y_not_in_x=torch.isin(neigh_y_list,neigh_list,invert=True) #pred中未被预测出的邻居

                    fp_neigh=neigh_list[x_not_in_y]
                    fn_neigh=neigh_y_list[y_not_in_x]

                    if len(fp_neigh)>0:
                        for fp in fp_neigh:

                            if fp in labels_y:
                                fp_mask[torch.where((vessel_==lab) & (y_vessel==fp))]=1
                            else:
                                fp_mask[torch.where(vessel_==fp)]=1
                    
                    if len(fn_neigh)>0:
                        for fn in fn_neigh:
                            if fn not in labels_y:
                                continue
                            
                            aa=torch.zeros_like(y_vessel,dtype=torch.bool)
                            aa[torch.where(y_vessel==lab)]=1
                            bb=torch.zeros_like(y_vessel,dtype=torch.bool)
                            bb[torch.where(vessel_==lab)]=1
                            fn_mask[torch.where(torch.logical_xor(aa,bb))]=1
                        
                            aa=torch.zeros_like(y_vessel,dtype=torch.bool)
                            aa[torch.where(y_vessel==fn)]=1
                            bb=torch.zeros_like(y_vessel,dtype=torch.bool)
                            bb[torch.where(vessel_==fn)]=1
                            fn_mask[torch.where(torch.logical_xor(aa,bb))]=1

            fp_mask_all[b]=fp_mask
            fn_mask_all[b]=fn_mask

        return fp_mask_all,fn_mask_all
    
    def FP_adjcency_coef(self,x,y_onehot):
        kernels=torch.ones((y_onehot.shape[1],1,3,3,3),device=x.device)
        y_onehot=y_onehot.float()
        y_connect_onehot=F.conv3d(y_onehot,weight=kernels, padding=3//2,groups=y_onehot.shape[1])
        A_matrix=torch.zeros((x.shape[0],len(self.connection_list)+1,len(self.connection_list)+1),device=x.device)
        for b in range(x.shape[0]):
            for i in range(1,len(self.connection_list)+1):
                neigh_list=self.connection_list[i-1]
                for j in range(1,len(self.connection_list)+1):
                    if i==j:
                        continue
                    if j not in neigh_list:

                        tmp=(x[b,j-1]-x[b,i-1])*y_connect_onehot[b,i-1]
                        A_matrix[b,i,j]=torch.sum(tmp)

        return A_matrix
    
    def FN_adjcency_coef(self,x,y_onehot,FN_mask):
        y_FN=y_onehot*FN_mask
        x_cla=x.argmax(1)[:, None]
        print('x_cla.shape:',x_cla.shape)

        A_matrix=torch.zeros((x.shape[0],len(self.connection_list)+1,len(self.connection_list)+1),device=x.device)
        for b in range(x.shape[0]):
            for i in range(1,len(self.connection_list)+1):
                y_i=y_FN[b,i-1].unsqueeze(0)
                print('y_i:',y_i.shape)
                x_shouldbe_i=torch.unique(x_cla[b][y_i>0])
                print('x_shouldbe_i:',x_shouldbe_i)
                for j in x_shouldbe_i:
                    tmp=(x[b,j-1]-x[b,i-1])
                    print('tmp.shape:',tmp.shape)
                    tmp[x_cla[b,0]!=j-1]=0
                    A_matrix[b,i,j]=torch.sum(tmp)# need to think carefully

        return A_matrix

    def forward(self, x, y, weight, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        x_class=x.argmax(1)[:, None]

        if not self.do_bg:
            x = x[:, 1:]


        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))
                

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y

                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]


        fp_mask,fn_mask=self.get_fp_fn(x_class,y,21)
        intersect = (x * y_onehot *weight).sum(axes) if loss_mask is None else (x * y_onehot * weight* loss_mask).sum(axes)

        FP_neigh=(x*fp_mask*weight).sum(axes) if loss_mask is None else (x*fp_mask*weight* loss_mask).sum(axes)
        FN_neigh=(x*fn_mask*weight).sum(axes) if loss_mask is None else (x*fn_mask*weight* loss_mask).sum(axes)


        
        
        if self.ddp and self.batch_dice:

            intersect = AllGatherGrad.apply(intersect).sum(0)
            FP_neigh = AllGatherGrad.apply(FP_neigh).sum(0)
            FN_neigh = AllGatherGrad.apply(FN_neigh).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            FP_neigh = FP_neigh.sum(0)
            FN_neigh = FN_neigh.sum(0)


        dc = (2 * intersect + self.smooth) / (torch.clip(2 * intersect +FP_neigh +FN_neigh + self.smooth, 1e-8))
        dc=dc.mean() 

        return -dc
         

    

    def forward_add(self, x, y, weight, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        x_class=x.argmax(1)[:, None]

        if not self.do_bg:
            x = x[:, 1:]


        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))


        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))
                

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y

                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)


            
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]


        fp_mask,fn_mask=self.get_fp_fn(x_class,y,21)


        FP_coef=self.FP_adjcency_coef(x,y_onehot)
        FP_coef=FP_coef.sum(dim=2)
        FP_coef=FP_coef[:,1:]
        
        FN_coef=self.FN_adjcency_coef(x,y_onehot,fn_mask)
        FN_coef=FN_coef.sum(dim=2)
        FN_coef=FN_coef[:,1:]


        intersect = (x * y_onehot *weight).sum(axes) if loss_mask is None else (x * y_onehot * weight* loss_mask).sum(axes)

        FP_neigh=(x*fp_mask*weight).sum(axes) if loss_mask is None else (x*fp_mask*weight* loss_mask).sum(axes)
        FN_neigh=(x*fn_mask*weight).sum(axes) if loss_mask is None else (x*fn_mask*weight* loss_mask).sum(axes)
    

        
        
        
        if self.ddp and self.batch_dice:

            intersect = AllGatherGrad.apply(intersect).sum(0)
            FP_neigh = AllGatherGrad.apply(FP_neigh).sum(0)
            FN_neigh = AllGatherGrad.apply(FN_neigh).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            FP_neigh = FP_neigh.sum(0)
            FN_neigh = FN_neigh.sum(0)


        dc = (2 * intersect + self.smooth) / (torch.clip(2 * intersect +FP_neigh +FN_neigh + self.smooth, 1e-8))
        

        add=FP_coef+FN_coef

        dc=dc+add
        
        dc=dc.mean() 


        return -dc
    
    def forward_multiply(self, x, y, weight, loss_mask=None):
        # y=tube skel for label
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        x_class=x.argmax(1)[:, None]

        if not self.do_bg:
            x = x[:, 1:]

        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
                # skel = skel.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))
                

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y

                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
                

        fp_mask,fn_mask=self.get_fp_fn(x_class,y,21)
        intersect = (x * y_onehot *weight).sum(axes) if loss_mask is None else (x * y_onehot * weight* loss_mask).sum(axes)

        FP_neigh=(x*fp_mask*weight).sum(axes) if loss_mask is None else (x*fp_mask*weight* loss_mask).sum(axes)
        FN_neigh=(x*fn_mask*weight).sum(axes) if loss_mask is None else (x*fn_mask*weight* loss_mask).sum(axes)

        FP_coef=self.FP_adjcency_coef(x,y_onehot)
        FP_coef=FP_coef.sum(dim=2)
        FP_coef=FP_coef[:,1:]
        FN_coef=self.FN_adjcency_coef(x,y_onehot,fn_mask)
        FN_coef=FN_coef.sum(dim=2)
        FN_coef=FN_coef[:,1:]
        FP_coef=torch.clip(FP_coef + self.smooth, 1e-8)
        FN_coef=torch.clip(FN_coef + self.smooth, 1e-8)

        FP_neigh=FP_neigh*FP_coef
        FN_neigh=FN_neigh*FN_coef
        
        
        if self.ddp and self.batch_dice:

            intersect = AllGatherGrad.apply(intersect).sum(0)
            FP_neigh = AllGatherGrad.apply(FP_neigh).sum(0)
            FN_neigh = AllGatherGrad.apply(FN_neigh).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            FP_neigh = FP_neigh.sum(0)
            FN_neigh = FN_neigh.sum(0)


        dc = (2 * intersect + self.smooth) / (torch.clip(2 * intersect +FP_neigh +FN_neigh + self.smooth, 1e-8))
        dc=dc.mean() 


        return -dc
    

# adjacency-aware dice loss (differentiable version, modified after MICCAI rebuttal)
class AdjacencyLoss_Weight(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(AdjacencyLoss_Weight, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        # self.use_add=use_add

        self.connection_list=[[2,3,15,16],
                [1,19,8],
                [1,20,9],
                [5,8,11,13],
                [4],
                [7,9,12,14],
                [6],
                [2,4,19],
                [3,6,20],
                [11,12,17,18],
                [4,10,17],
                [6,10,18],
                [4],
                [6],
                [1],
                [1],
                [10,11],
                [10,12],
                [2,8],
                [3,9]]
        # 3 classes:spacial near/morology similar/not similar
        self.nonConnection_list=[[[19,20],[4,6],[5,7,8,9,10,11,12,13,14,17,18]],
                                 [[15,13],[3,11,12,16,20],[4,5,6,7,8,9,10,14,17,18]],
                                 [[16,14],[2,11,12,15,19],[4,5,6,7,8,9,10,13,17,18]],
                                 [[10,17],[6,1],[2,3,7,9,12,14,15,16,18,19,20]],
                                 [[11,8,13],[7,2],[1,3,6,9,10,12,14,15,16,17,18,19,20]],
                                 [[10,18],[4,1],[2,3,5,8,11,13,15,16,17,19,20]],
                                 [[12,9,14],[5,3],[1,2,4,8,10,11,13,15,16,17,18,19,20]],
                                 [[13,11],[9,15],[1,3,5,6,7,10,12,14,16,17,18,20]],
                                 [[14,12],[8,16],[1,2,4,5,7,10,11,13,15,17,18,19,20]],
                                 [[13,14],[8,9],[1,2,3,4,5,6,7,15,16,19,20]],
                                 [[5,8,13],[12,18,2,3],[1,6,7,9,14,15,16,19,20]],
                                 [[7,9,14],[11,17,2,3],[1,4,5,8,13,15,16,19,20]],
                                 [[8,2,19],[14,15],[1,3,5,6,7,9,10,11,12,16,17,18,20]],
                                 [[9,3,20],[13,16],[1,2,4,5,7,8,10,11,12,15,17,18,19]],
                                 [[2,19],[16,13],[3,4,5,6,7,8,9,10,11,12,14,17,18,20]],
                                 [[3,20],[15,14],[2,4,5,6,7,8,9,10,11,12,13,17,18,19]],
                                 [[18,4],[1,12],[2,3,5,6,7,8,9,13,14,15,16,19,20]],
                                 [[17,6],[1,11],[2,3,4,5,7,8,9,13,14,15,16,19,20]],
                                 [[1,15],[13,20],[3,4,5,6,7,9,10,11,12,14,16,17,18]],
                                 [[1,16],[14,19],[2,4,5,6,7,8,10,11,12,13,15,17,18]]
                                ]
        


        self.weights=[1.5,1.5,1]
        self.is_merge_list=True

        if self.is_merge_list:
            self.nonConnection_list=self.merge_list()
        

    def merge_list(self):
        non_adjacency_list=[]
        for i in range(len(self.nonConnection_list)):
            list_i=self.nonConnection_list[i]
            list_tmp=[]
            for list_ in list_i:
                # print('i:',i)
                # print(list_)
                list_tmp.extend(list_)
            non_adjacency_list.append(list_tmp)

        return non_adjacency_list

    def get_nonConnection_matrix(self,class_number):
        non_adj_matrix=torch.zeros((class_number,class_number))
        if class_number==20:
            for i in range(class_number):
                if self.is_merge_list:
                    for j in self.nonConnection_list[i]:
                        non_adj_matrix[i,j-1]=1
                else:
                    for k,list_ in enumerate(self.nonConnection_list[i]):
                        for j in list_:
                            non_adj_matrix[i,j-1]=1*self.weights[k]
        else:# when use at topcow,we can expand it
            raise RuntimeError('class number is wrong!')
        
        return non_adj_matrix

    def get_Connection_matrix(self,class_number,y_unique):
        adj_matrix=torch.zeros((class_number,class_number))
        if class_number==20:
            for i in range(class_number):
                if i not in y_unique:
                    continue
                for j in self.connection_list[i]:
                    if j not in y_unique:
                        continue
                    adj_matrix[i,j-1]=1
                
        else:# when use at topcow,we can expand it
            raise RuntimeError('class number is wrong!')
        
        return adj_matrix
    

    def FP_component_withY(self,x,y):

        FP_dc=torch.zeros((x.shape[0],20),device=x.device)
        for b in range(x.shape[0]):
            y_unique=torch.unique(y[b])
            for i in y_unique:
                if i==0:
                    continue
                x_i=x[b,i]# we suppose the x is without bg
                fp_dc_sum=0
                for non_list,alpha in zip(self.nonConnection_list[i-1],self.weights):
                    x_neigh=x[b,non_list]
                    shp=x_neigh.shape
                    kernel=torch.ones((shp[0],3,3,3),device=x.device)
                    neigh=F.conv3d(x_neigh,weight=kernel, padding=1,groups=shp[0])
                    x_i=x_i.unsqueeze(0)
                    intersect=(x_i*neigh).sum([1,2,3])
                    sum_x=(x_i**2).sum([1,2,3])
                    sum_neigh=(neigh**2).sum([1,2,3])
                    fp_dc=(2*intersect+self.smooth)/(sum_x+sum_neigh+self.smooth)
                    fp_dc=(fp_dc*alpha).sum()
                    fp_dc_sum+=fp_dc
                FP_dc[b,i]=fp_dc_sum

        FP_dc=FP_dc.mean()
        return FP_dc
    
    def FP_component_withoutY(self,x,weight=None):

        FP_dc=torch.zeros((20),device=x.device)
        for i in range(21):
            if i==0:
                continue
            x_i=x[:,i]# we suppose the x is with bg
            fp_dc_sum=0
            if self.is_merge_list:
                
                x_neigh=x[:,self.nonConnection_list[i-1]]
                shp=x_neigh.shape
                kernel=torch.ones((shp[1],1,3,3,3),device=x.device)
                neigh=F.conv3d(x_neigh,weight=kernel, padding=1,groups=shp[1])
                x_i=x_i.unsqueeze(1)
                intersect=(x_i*neigh*weight).sum([1,2,3])
                sum_x=(x_i**2*weight).sum([1,2,3])
                sum_neigh=(neigh**2*weight).sum([1,2,3])
                fp_dc=(2*intersect+self.smooth)/(sum_x+sum_neigh+self.smooth)
                fp_dc=(fp_dc).sum()
                fp_dc_sum+=fp_dc


            else:
                
                for non_list,alpha in zip(self.nonConnection_list[i-1],self.weights):
                    x_neigh=x[:,non_list]
                    shp=x_neigh.shape
                    neigh=F.conv3d(x_neigh,weight=kernel, padding=1,groups=shp[1])
                    x_i=x_i.unsqueeze(1)
                    intersect=(x_i*neigh*weight).sum([1,2,3])
                    sum_x=(x_i**2*weight).sum([1,2,3])
                    sum_neigh=(neigh**2*weight).sum([1,2,3])
                    fp_dc=(2*intersect+self.smooth)/(sum_x+sum_neigh+self.smooth)
                    fp_dc=(fp_dc*alpha).sum()
                    fp_dc_sum+=fp_dc

            FP_dc[i-1]=fp_dc_sum
        FP_dc=FP_dc.sum()
        return FP_dc

    def FP_componentV2_withoutY(self,x,weight=None):
        
        B,C,D,H,W=x.shape
        axes = list(range(2, len(x.shape)))
        kernel=torch.ones((C,1,3,3,3),device=x.device)
        kernel[:,0,1,1,1]=0
        
        neigh=F.conv3d(x,weight=kernel, padding=(1,1,1),groups=C)
        neigh=neigh/26

        if weight is not None:
            x=x*weight
        with torch.no_grad():

            self.nonConnection_matrix=torch.zeros((B,C,C),device=x.device)
            for b in range(B):

                self.nonConnection_matrix[b]=self.get_nonConnection_matrix(C).to(x.device)


        x_flat=x.view(B,C,-1)
        neigh_flat=neigh.view(B,C,-1)

        intersect=torch.einsum("bij,bjk->bik",x_flat,neigh_flat.transpose(1,2))

        if torch.isnan(intersect).all():
            print('intersect get all nan!need to check')
            return 1e-5

        print('if nan in intersect:',torch.isnan(intersect).any())
        print('if inf in intersect:',torch.isinf(intersect).any())
        if torch.isinf(intersect).any():
            print('intersect:',intersect)

        x_vol=torch.sum(x.pow(2),dim=axes)
        if weight is not None:
            neigh=neigh*weight
        neigh_vol=torch.sum(neigh.pow(2),dim=axes)

        denominator=torch.clip(x_vol.unsqueeze(2)+neigh_vol.unsqueeze(1),min=1e-8,max=1e7)
        denominator=torch.nan_to_num(denominator,nan=0.0,posinf=1e5,neginf=-1e5)

        intersect=intersect*self.nonConnection_matrix
        denominator=denominator*self.nonConnection_matrix


        nonAdj_dc=(2.0*intersect+self.smooth)/(torch.clip(denominator+ self.smooth, 1e-8))

        if torch.isnan(nonAdj_dc).any() or torch.isinf(nonAdj_dc).any():
            print('nonAdj_dc:',nonAdj_dc)
        if torch.isnan(nonAdj_dc).any():
            dc_min=torch.min(nonAdj_dc[~torch.isnan(nonAdj_dc)])
            print('where is nan?',torch.nonzero(torch.isnan(nonAdj_dc)))
            nonAdj_dc=torch.where(torch.isnan(nonAdj_dc),dc_min,nonAdj_dc)

        if torch.isinf(nonAdj_dc).any():
            dc_max=torch.max(nonAdj_dc[~torch.isinf(nonAdj_dc)])
            print('dc_max:',dc_max)
            nonAdj_dc=torch.where(torch.isinf(nonAdj_dc),dc_max,nonAdj_dc)


        if self.ddp and self.batch_dice:
            nonAdj_dc = AllGatherGrad.apply(nonAdj_dc).sum(0)
 
        if self.batch_dice:
            print('batch dice= true.')
            nonAdj_dc = nonAdj_dc.sum(0)

        dc_loss=torch.clip(torch.sum(nonAdj_dc),0)/(torch.clip(self.nonConnection_matrix.sum()*B+ self.smooth, 1e-8))

        return dc_loss

    def FN_componentV2_withY(self,x,y,weight=None):

        x_bg=x[:,0]
        x_bg=x_bg*weight
        bg_prohibit=x_bg[x_bg>0].mean()
            
        x=x[:,1:]

        B,C,D,H,W=x.shape

        with torch.no_grad():
            self.Connection_matrix=torch.zeros((B,C,C),device=x.device)
            for b in range(B):
                y_unique=torch.unique(y[b])
                self.Connection_matrix[b]=self.get_Connection_matrix(C,y_unique).to(x.device)

        axes = list(range(2, len(x.shape)))
        kernel=torch.ones((C,1,3,3,3),device=x.device)
        kernel[:,0,1,1,1]=0
        
        neigh=F.conv3d(x,weight=kernel, padding=(1,1,1),groups=C)
        neigh=neigh/26

        if weight is not None:
            x=x*weight
        x_flat=x.view(B,C,-1)
        neigh_flat=neigh.view(B,C,-1)

        intersect=torch.einsum("bij,bjk->bik",x_flat,neigh_flat.transpose(1,2))
        intersect=torch.nan_to_num(intersect,nan=0.0,posinf=1e5,neginf=-1e5)
        

        x_vol=torch.sum(x.pow(2),dim=axes)
        if weight is not None:
            neigh=neigh*weight
        neigh_vol=torch.sum(neigh.pow(2),dim=axes)

        denominator=x_vol.unsqueeze(2)+neigh_vol.unsqueeze(1)
        denominator=torch.nan_to_num(denominator,nan=0.0,posinf=1e5,neginf=-1e5)

        dc=(2.0*intersect+self.smooth)/torch.clip(denominator+self.smooth,self.smooth)

        Adj_dc=dc*self.Connection_matrix

        if self.ddp and self.batch_dice:
            Adj_dc = AllGatherGrad.apply(Adj_dc).sum(0)
 
        if self.batch_dice:
            Adj_dc = Adj_dc.sum(0)


        dc_loss=torch.sum(Adj_dc)/torch.clip(self.Connection_matrix.sum()+ self.smooth, 1e-8)
        dc_loss=-0.1*dc_loss+0.5*bg_prohibit

        return dc_loss
        

    def FN_component_withY(self,x,y,weight=None):
        kernel=torch.ones((1,3,3,3),device=x.device)
        FN_dc=torch.zeros((x.shape[0],20),device=x.device)
        for b in range(x.shape[0]):
            y_unique=torch.unique(y[b])
            for i in y_unique:
                if i==0:
                    continue
                x_i=x[b,i]# we suppose the x is without bg

                adj_list_tmp=self.connection_list[i-1]
                adj_list=list(set(y_unique) & set(adj_list_tmp))
                yi_bin=torch.where(y[b]==i,1,0)

                yi_bin=yi_bin.unsqueeze(0)
                neigh_po=torch.isin(y,torch.tensor(adj_list,device=x.device))
                neigh_bin=torch.where(neigh_po,1,0)

                dilate_lab_vessel=self.soft_dilate(yi_bin)
                dilate_lab_vessel=dilate_lab_vessel.int()
                dilate_lab_vessel=dilate_lab_vessel.squeeze(0)

                neigh_vessel=yi_bin*dilate_lab_vessel # 找pred当前类别的邻居
                neigh_vessel

            
                x_neigh=x[b,adj_list]
                shp=x_neigh.shape
                neigh=F.conv3d(x_neigh,weight=kernel, padding=1,groups=shp[0])
                x_i=x_i.unsqueeze(0)
                intersect=(x_i*neigh*weight).sum([1,2,3])
                sum_x=(x_i**2*weight).sum([1,2,3])
                sum_neigh=(neigh**2*weight).sum([1,2,3])
                fn_dc=(2*intersect+self.smooth)/(sum_x+sum_neigh+self.smooth)
                fn_dc=(fn_dc*alpha).sum()
                FN_dc[b,i]=fn_dc

        # FN_dc=FN_dc.mean()
        FN_dc=-FN_dc.sum()
        return FN_dc

    def forward(self, x, y, weight,FN_weight, loss_mask=None):
        # y=tube skel for label
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))

            if len(shp_x) != len(FN_weight.shape):
                FN_weight = FN_weight.view((FN_weight.shape[0], 1, *FN_weight.shape[1:]))

        FN_dc=self.FN_componentV2_withY(x,y,weight=FN_weight)

        if not self.do_bg:
            x = x[:, 1:]

        FP_dc=self.FP_componentV2_withoutY(x,weight=weight)

        if FP_dc==1e-5:
            print('if nan is from original x?',torch.isnan(x).all())

        return FP_dc+FN_dc


# adjacency-aware co-occurrence loss (journal version)
class AdjacencyCoOccurenceLoss_Weight(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(AdjacencyCoOccurenceLoss_Weight, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

        self.connection_list=[[2,3,15,16],
                [1,19,8],
                [1,20,9],
                [5,8,11,13],
                [4],
                [7,9,12,14],
                [6],
                [2,4,19],
                [3,6,20],
                [11,12,17,18],
                [4,10,17],
                [6,10,18],
                [4],
                [6],
                [1],
                [1],
                [10,11],
                [10,12],
                [2,8],
                [3,9]]
        # nonConnection_list with 3 classes:spacial near/morology similar/not similar
        self.nonConnection_list=[[[19,20],[4,6],[5,7,8,9,10,11,12,13,14,17,18]],
                                 [[15,13],[3,11,12,16,20],[4,5,6,7,8,9,10,14,17,18]],
                                 [[16,14],[2,11,12,15,19],[4,5,6,7,8,9,10,13,17,18]],
                                 [[10,17],[6,1],[2,3,7,9,12,14,15,16,18,19,20]],
                                 [[11,8,13],[7,2],[1,3,6,9,10,12,14,15,16,17,18,19,20]],
                                 [[10,18],[4,1],[2,3,5,8,11,13,15,16,17,19,20]],
                                 [[12,9,14],[5,3],[1,2,4,8,10,11,13,15,16,17,18,19,20]],
                                 [[13,11],[9,15],[1,3,5,6,7,10,12,14,16,17,18,20]],
                                 [[14,12],[8,16],[1,2,4,5,7,10,11,13,15,17,18,19,20]],
                                 [[13,14],[8,9],[1,2,3,4,5,6,7,15,16,19,20]],
                                 [[5,8,13],[12,18,2,3],[1,6,7,9,14,15,16,19,20]],
                                 [[7,9,14],[11,17,2,3],[1,4,5,8,13,15,16,19,20]],
                                 [[8,2,19],[14,15],[1,3,5,6,7,9,10,11,12,16,17,18,20]],
                                 [[9,3,20],[13,16],[1,2,4,5,7,8,10,11,12,15,17,18,19]],
                                 [[2,19],[16,13],[3,4,5,6,7,8,9,10,11,12,14,17,18,20]],
                                 [[3,20],[15,14],[2,4,5,6,7,8,9,10,11,12,13,17,18,19]],
                                 [[18,4],[1,12],[2,3,5,6,7,8,9,13,14,15,16,19,20]],
                                 [[17,6],[1,11],[2,3,4,5,7,8,9,13,14,15,16,19,20]],
                                 [[1,15],[13,20],[3,4,5,6,7,9,10,11,12,14,16,17,18]],
                                 [[1,16],[14,19],[2,4,5,6,7,8,10,11,12,13,15,17,18]]
                                ]
        


        self.weights=[1.5,1.5,1]# weight for 3 classes
        self.is_merge_list=True

        if self.is_merge_list:
            self.nonConnection_list=self.merge_list()
        

    def merge_list(self):
        non_adjacency_list=[]
        for i in range(len(self.nonConnection_list)):
            list_i=self.nonConnection_list[i]
            list_tmp=[]
            for list_ in list_i:
                list_tmp.extend(list_)
            non_adjacency_list.append(list_tmp)

        return non_adjacency_list

    def get_nonConnection_matrix(self,class_number):
        # return the nonConnection matrix(C,C),if (i,j) is not connected, non_adj_matrix[i,j]=weights, else non_adj_matrix[i,j]=0
        non_adj_matrix=torch.zeros((class_number,class_number))
        if class_number==20:
            for i in range(class_number):
                if self.is_merge_list:
                    for j in self.nonConnection_list[i]:
                        non_adj_matrix[i,j-1]=1
                else:
                    for k,list_ in enumerate(self.nonConnection_list[i]):
                        for j in list_:
                            non_adj_matrix[i,j-1]=1*self.weights[k]
        else:# when use at topcow,we can expand it
            raise RuntimeError('class number is wrong!')
        
        return non_adj_matrix

    def get_Connection_matrix(self,class_number,y_unique):
        # return the connection matrix(C,C),if (i,j) is connected, adj_matrix[i,j]=1, else adj_matrix[i,j]=0
        adj_matrix=torch.zeros((class_number,class_number))
        if class_number==20:
            for i in range(class_number):
                if i not in y_unique:
                    continue
                for j in self.connection_list[i]:
                    if j not in y_unique:
                        continue
                    adj_matrix[i,j-1]=1
                
        else:# when use at topcow,we can expand it
            raise RuntimeError('class number is wrong!')
        
        return adj_matrix


    def FP_CoOcurrence_withY(self,x,y,weight=None):

        # x except bg
        B,C,D,H,W=x.shape
        if weight is not None:
            x=x*weight
        x_avg=F.avg_pool3d(x,kernel_size=3,stride=1,padding=1)
        print(x_avg.shape)

        with torch.no_grad():

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros((B,C+1,D,H,W), device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]
                y_avg=F.avg_pool3d(y_onehot,kernel_size=3,stride=1,padding=1)
                if weight is not None:
                    y_avg=y_avg*weight
        

        local_x=x_avg.unsqueeze(2)*y_avg.unsqueeze(1)

        with torch.no_grad():


            self.nonConnection_matrix=torch.zeros((B,C,C),device=x.device)
            for b in range(B):

                self.nonConnection_matrix[b]=self.get_nonConnection_matrix(C).to(x.device)

            self.nonConnection_matrix=self.nonConnection_matrix.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        penal_cooccurrence=local_x*self.nonConnection_matrix
        del local_x

        sum_gt=y_avg.sum(dim=(2,3,4))
        sum_x=x_avg.sum(dim=(2,3,4))



        cooccur=penal_cooccurrence.sum(dim=(2,3,4,5))
        print(penal_cooccurrence.sum(dim=(1,2)).mean())

        cooccur_loss=cooccur/torch.clip(sum_gt+sum_x+self.smooth,1e-8)

        cooccur_loss=cooccur_loss.mean()
        

        return cooccur_loss


    def FP_CoOcurrence(self,x,weight=None):

        # x except bg
        B,C,D,H,W=x.shape
        if weight is not None:
            x=x*weight
        # x_avg=F.avg_pool3d(x,kernel_size=3,stride=1,padding=1)
        kernels=torch.ones((B,1,3,3,3),device=x.device)
        x_avg=F.conv3d(x,weight=kernels, padding=1,groups=C) # extract the co-occurence feature
        # print(x_avg.shape) #(B,C,D,H,W)

        
        # Make each image of a category (channel) multiply with the image of every other category (channel)
        local_x=x_avg.unsqueeze(2)*x_avg.unsqueeze(1) #(B,C,1,D,H,W)*(B,1,C,D,H,W) 
        # （B,C,C,D,H,W)
        with torch.no_grad():
            self.nonConnection_matrix=torch.zeros((B,C,C),device=x.device)
            for b in range(B):
                # Extract the non-adjacent categories within each category, and set the adjacent categories to 0.
                self.nonConnection_matrix[b]=self.get_nonConnection_matrix(C).to(x.device) 

            self.nonConnection_matrix=self.nonConnection_matrix.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #（B,C,C,1,1,1)

        # Calculate the co-occurrence penalty. The weights of adjacent categories are 0, and the penalty value is 0.
        penal_cooccurrence=local_x*self.nonConnection_matrix 
        del local_x

        sum_x=x_avg.sum(dim=(2,3,4))#(B,C)

        cooccur=penal_cooccurrence.sum(dim=(2,3,4,5))#(B,C)

        cooccur_loss=cooccur/torch.clip(sum_x+self.smooth,1e-8) #penalty value/all co-occurence value
        
        return cooccur_loss



    def FN_componentV3_withY(self,x,y,weight=None):
        # weight: kerpoint map

        x_bg=x[:,0]
        if weight is not None:
            x_bg=x_bg*weight
        bg_prohibit=x_bg[x_bg>0].mean() #suppress the false negative in background
        # print('bg_prohibit:',bg_prohibit)
     
        with torch.no_grad():

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]

        x=x[:,1:]
        B,C,W,H,D=x.shape

        with torch.no_grad():
            # kernel of sobel
            kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=x.device)
            kernel_y = kernel_x.rot90(1, dims=(1, 3)) # rotated in H,W
            kernel_z = kernel_x.rot90(1, dims=(2, 3)) # rotated in H,D

            # Transform the kernel into the format of [out_channels, in_channels, D, H, W] for the convolution kernel.
            self.sobel_x = kernel_x.view(1, 1, 3, 3, 3)
            self.sobel_y = kernel_y.view(1, 1, 3, 3, 3)
            self.sobel_z = kernel_z.view(1, 1, 3, 3, 3)

            sobel_x=self.sobel_x.repeat(C,1,1,1,1)
            sobel_y=self.sobel_y.repeat(C,1,1,1,1)
            sobel_z=self.sobel_z.repeat(C,1,1,1,1)

        if weight is not None:
            x=x*weight
            y=y*weight

        # Perform Sobel convolution on each category within the keypoint area to extract the intersection edges of each blood vessel.
        grad_x=F.conv3d(x,sobel_x,stride=1,padding='same',groups=C)
        grad_y=F.conv3d(x,sobel_y,stride=1,padding='same',groups=C)
        grad_z=F.conv3d(x,sobel_z,stride=1,padding='same',groups=C)


        gt_grad_x=F.conv3d(y_onehot,sobel_x,stride=1,padding='same',groups=C)
        gt_grad_y=F.conv3d(y_onehot,sobel_y,stride=1,padding='same',groups=C)
        gt_grad_z=F.conv3d(y_onehot,sobel_z,stride=1,padding='same',groups=C)

        mag=torch.sqrt(grad_x**2+grad_y**2+grad_z**2+1e-8)

        gt_mag=torch.sqrt(gt_grad_x**2+gt_grad_y**2+gt_grad_z**2+1e-8)

        mse_loss=F.mse_loss(mag,gt_mag)# MSE loss
        

        return 0.1*(mse_loss+bg_prohibit)
    

    def forward(self,x,y,weight=None,FN_weight=None, loss_mask=None):

        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))

            if len(shp_x) != len(FN_weight.shape):
                FN_weight = FN_weight.view((FN_weight.shape[0], 1, *FN_weight.shape[1:]))

        FN_dc=self.FN_componentV3_withY(x,y,weight=FN_weight)

        if not self.do_bg:
            x = x[:, 1:]

        FP_dc=self.FP_CoOcurrence(x,weight=weight)


        return FP_dc+FN_dc


# radius-aware CE loss (abandoned)
class Weight_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(Weight_CrossEntropyLoss,self).__init__()


    def forward(self,x,y,weight):
        if len(y.shape) == len(x.shape):
            assert y.shape[1] == 1
            y = y[:, 0]
        if len(weight.shape) == len(x.shape):
            assert weight.shape[1] == 1
            weight = weight[:, 0]
        y=y.long()
        l_ce=F.cross_entropy(x,y,reduction='none')
        ce_weight=torch.mul(l_ce,weight)
        return ce_weight.mean()

# breakage-aware MSE loss (MICCAI version)
class Weight_CubeConnectCELoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, kernel_size: int=3, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(Weight_CubeConnectCELoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.kernel_size=kernel_size
        


    def forward(self, x, y, weight, loss_mask=None):
        # y=tube skel for label
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        x_class=x.argmax(1)[:, None]
        x_onehot = torch.zeros(shp_x, device=x.device)
        x_onehot.scatter_(1, x_class, 1)

        if not self.do_bg:
            x = x[:, 1:]
            x_onehot=x_onehot[:,1:]


        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))


        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
                # skel = skel.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
                # skel_onehot=skel
                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

                y_vessel=torch.where(y>0,1,0).bool()
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
        y_onehot=y_onehot.float()

        kernels=torch.ones((y_onehot.shape[1],1,self.kernel_size,self.kernel_size,self.kernel_size),device=x.device)

        label_connect_onehot=F.conv3d(y_onehot,weight=kernels, padding=self.kernel_size//2,groups=y_onehot.shape[1])
        x_fore_prob=x*x_onehot
        pred_connect_onehot=F.conv3d(x_fore_prob,weight=kernels, padding=self.kernel_size//2,groups=x.shape[1])

        loss_=F.mse_loss(pred_connect_onehot, label_connect_onehot,reduction='none')


        loss_=loss_*weight*y_vessel
        
        loss_=loss_.mean()


        return loss_
    

from nnunetv2.training.loss.skeletonize_cbdice import Skeletonize
from nnunetv2.training.loss.soft_skeleton_cbdice import SoftSkeletonize
# breakage-aware clDice loss (journal version)
class SoftNeighErrorMulticlDiceLoss(torch.nn.Module):
    def __init__(self, apply_nonlin: Callable = None, kernel_size: int=3, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, iter_=10):
        super(SoftNeighErrorMulticlDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.kernel_size=kernel_size

    def forward(self, y_pred, y_true,weight=None, t_skeletonize_flage=False,loss_mask=None):
        
        shp_x, shp_y = y_pred.shape, y_true.shape

        axes = list(range(2, len(shp_x)))


        if self.apply_nonlin is not None:
            y_pred_fore = y_pred[:, 1:]# extract the foreground
            y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
            y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1) #concat with background
          
            y_prob_binary = self.apply_nonlin(y_pred_binary)# softmax
            y_pred_prob = y_prob_binary[:, 1] #prob of foreground

            y_pred = self.apply_nonlin(y_pred)

        if not self.do_bg:
            y_pred = y_pred[:, 1:]
        

        with torch.no_grad():

            if len(shp_x) != len(shp_y):
                y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))
            
            if weight is not None:
                shp_w=weight.shape
                if len(shp_x) != len(shp_w):
                    weight = weight.view((shp_w[0], 1, *shp_w[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y_true

                print('1')
            else:
                gt = y_true.long()
                y_onehot = torch.zeros(shp_x, device=y_pred.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            y_onehot=y_onehot.float()


            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float() # binary pred
        
            # extract the skeleton using foreground pred
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:

                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1))
                skel_true = self.m_skeletonize(y_true.unsqueeze(1))

        
        skel_pred_prob = skel_pred_hard * y_pred # skeleton for all channels
        skel_true_oh = skel_true * y_onehot

        # topolpgy error map
        kernels=torch.ones((y_onehot.shape[1],1,self.kernel_size,self.kernel_size,self.kernel_size),device=y_pred.device) #conv kernel
        label_connect_onehot=F.conv3d(y_onehot,weight=kernels, padding=self.kernel_size//2,groups=y_onehot.shape[1])        
        pred_connect_onehot=F.conv3d(y_pred,weight=kernels, padding=self.kernel_size//2,groups=y_pred.shape[1])
        error_map=F.smooth_l1_loss(pred_connect_onehot, label_connect_onehot,reduction='none') #MAE error map

        error_map=torch.exp(error_map/27) # non-linear func

        # add error map into cldice loss
        if weight is not None:
            tprec = (torch.sum(torch.multiply(skel_pred_prob, y_onehot)*error_map*weight,dim=axes)+self.smooth)/(torch.sum(skel_pred_prob*error_map*weight,dim=axes)+self.smooth)    
            tsens = (torch.sum(torch.multiply(skel_true_oh, y_pred)*error_map*weight,dim=axes)+self.smooth)/(torch.sum(skel_true_oh*error_map*weight,dim=axes)+self.smooth)
        else:
            tprec = (torch.sum(torch.multiply(skel_pred_prob, y_onehot)*error_map,dim=axes)+self.smooth)/(torch.sum(skel_pred_prob*error_map,dim=axes)+self.smooth)    
            tsens = (torch.sum(torch.multiply(skel_true_oh, y_pred)*error_map,dim=axes)+self.smooth)/(torch.sum(skel_true_oh*error_map,dim=axes)+self.smooth)
        cl_dice_loss = - 2.0 * (tprec*tsens)/(tprec+tsens)
        cl_dice_loss=cl_dice_loss.mean()

        return cl_dice_loss
    
# breakage-aware clCE loss
class SoftNeighErrorMulticlCELoss(torch.nn.Module):
    def __init__(self, apply_nonlin: Callable = None, kernel_size: int=3, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, iter_=10):
        super(SoftNeighErrorMulticlCELoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)
        self.ce=nn.CrossEntropyLoss(reduction='none')

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.kernel_size=kernel_size

    def forward(self, y_pred, y_true,weight=None, t_skeletonize_flage=False,loss_mask=None):
        
        shp_x, shp_y = y_pred.shape, y_true.shape

        axes = list(range(2, len(shp_x)))

        if len(y_true.shape) == len(y_pred.shape):
            assert y_true.shape[1] == 1
            y_true_ = y_true[:, 0]

        ce_loss=self.ce(y_pred, y_true_.long())

        if self.apply_nonlin is not None:
            y_pred_fore = y_pred[:, 1:]
            y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
            y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
            
            
            y_prob_binary = self.apply_nonlin(y_pred_binary)
            y_pred_prob = y_prob_binary[:, 1]

            y_pred = self.apply_nonlin(y_pred)

        if not self.do_bg:
            y_pred = y_pred[:, 1:]
        

        with torch.no_grad():

            if len(shp_x) != len(shp_y):
                y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))

            if weight is not None:
                shp_w=weight.shape
                if len(shp_x) != len(shp_w):
                    weight = weight.view((shp_w[0], 1, *shp_w[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y_true

                print('1')
            else:
                gt = y_true.long()
                y_onehot = torch.zeros(shp_x, device=y_pred.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            y_onehot=y_onehot.float()


            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:

                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1))
                skel_true = self.m_skeletonize(y_true.unsqueeze(1))

        # skel_pred_prob = skel_pred_hard * y_pred_prob
        skel_pred_prob = skel_pred_hard * y_pred
        skel_true_oh = skel_true * y_onehot

        # topolpgy error map
        kernels=torch.ones((y_onehot.shape[1],1,self.kernel_size,self.kernel_size,self.kernel_size),device=y_pred.device)
        label_connect_onehot=F.conv3d(y_onehot,weight=kernels, padding=self.kernel_size//2,groups=y_onehot.shape[1])        
        pred_connect_onehot=F.conv3d(y_pred,weight=kernels, padding=self.kernel_size//2,groups=y_pred.shape[1])
        error_map=F.smooth_l1_loss(pred_connect_onehot, label_connect_onehot,reduction='none')

        error_map=torch.exp(error_map/27)


        ce_loss=ce_loss.unsqueeze(1)
        if weight is not None:
            tprec = (torch.multiply(skel_pred_prob, ce_loss)*error_map*weight).mean()
            tsens = (torch.multiply(skel_true_oh, ce_loss)*error_map*weight).mean()
        else:
            tprec = (torch.multiply(skel_pred_prob, ce_loss)*error_map).mean()
            tsens = (torch.multiply(skel_true_oh, ce_loss)*error_map).mean()
        cl_ce=tprec+tsens

        return cl_ce
 
# breakage-aware SkelRecall loss   
class SoftNeighErrorMultiSkelRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, kernel_size: int=3, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftNeighErrorMultiSkelRecallLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.kernel_size=kernel_size
        


    def forward(self, x, y, weight, loss_mask=None):
        # y=tube skel for label
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        if not self.do_bg:
            x = x[:, 1:]


        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))


        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
                # skel = skel.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]


        y_onehot=y_onehot.float()
        kernels=torch.ones((y_onehot.shape[1],1,self.kernel_size,self.kernel_size,self.kernel_size),device=x.device)
        label_connect_onehot=F.conv3d(y_onehot,weight=kernels, padding=self.kernel_size//2,groups=y_onehot.shape[1])        
        pred_connect_onehot=F.conv3d(x,weight=kernels, padding=self.kernel_size//2,groups=x.shape[1])
        error_map=F.smooth_l1_loss(pred_connect_onehot, label_connect_onehot,reduction='none')
        
 
        sum_gt = (y_onehot*weight*error_map).sum(axes) if loss_mask is None else (y_onehot * loss_mask*weight*error_map).sum(axes)

        inter_rec = (x * y_onehot*weight*error_map).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask*weight*error_map).sum(axes)

        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)
        
         

        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        rec = rec.mean()
        return -rec
    



from torch import nn, Tensor
# radius classification loss
class FocalLoss3D(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        3D multiclass Focal Loss
        
        参数:
            alpha (Tensor, optional): class weight, (num_classes,)
            gamma (float): default:2
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss3D, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                raise TypeError("alphashould be the type of torch.Tensor")
          

    def forward(self, inputs, targets):
        """

        para:
            inputs: (B, C, D, H, W)
            targets: (B, D, H, W)
        
        return:
            torch.Tensor: loss
        """
        if self.alpha is not None:
            self.alpha=self.alpha.to(inputs.device)
        if len(targets.shape) == len(inputs.shape):
            assert targets.shape[1] == 1
            targets = targets[:, 0]
        # flatten: 3D volume to 2D matrix
        batch_size, num_classes = inputs.shape[0], inputs.shape[1]
        inputs_flat = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)  # (B*D*H*W, C)
        targets_flat = targets.view(-1).long()  # (B*D*H*W,)

        # log softmax
        log_probs = F.log_softmax(inputs_flat, dim=1)
        
        # get the log prob for each class
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze()  # (B*D*H*W,)
        pt = log_pt.exp()  # 概率值

        # baseline loss
        focal_term = (1 - pt) ** self.gamma
        focal_loss = -focal_term * log_pt  # (1-pt)^gamma * CE

        # add class weight
        if self.alpha is not None:
            alpha = self.alpha.gather(0, targets_flat)  
            focal_loss = alpha * focal_loss

        
        focal_loss = focal_loss.view(batch_size, -1)  # (B, D*H*W)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.view_as(targets)



class DualConsistencySoftDiceLoss(nn.Module):
    def __init__(self, weight_1to2=0.5,weight_2to1=0.5,apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(DualConsistencySoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.weight_1to2=weight_1to2
        self.weight_2to1=weight_2to1

    def forward(self, x_main, x_raidus, loss_mask=None):
        dc_1to2=self.forward_one(x_raidus,x_main,loss_mask=loss_mask)
        dc_2to1=self.forward_one(x_main,x_raidus,loss_mask=loss_mask)

        dc=self.weight_1to2*dc_1to2+self.weight_2to1*dc_2to1
        return dc


    def forward_one(self, x, y, loss_mask=None):
        shp_x= x.shape
        # shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            y = self.apply_nonlin(y)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        y=y.argmax(1)[:, None]
        shp_y = y.shape
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
                # keyP_mask=keyP
                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
                # keyP_mask = keyP_mask[:, 1:]
            sum_gt = (y_onehot).sum(axes) if loss_mask is None else (y_onehot* loss_mask).sum(axes)

        
        
        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = (x).sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            intersect = AllGatherGrad.apply(intersect).sum(0)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        try:

            dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        except Exception as error:
            print(error)
            print('\n','>>>' * 20)
            print(traceback.print_exc())


        dc=-dc.mean()



        return dc  


class MemoryEfficientSoftDiceLoss_RadBranchWeight(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss_RadBranchWeight, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, weight, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            weight=self.apply_nonlin(weight)

        

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))
            
            weight_arg=weight.argmax(1)[:,None]

            weight=torch.zeros_like(y, device=x.device)

            
            dis=2-(weight_arg-1)/(shp_weight[1]-1)
            xx=torch.amax(dis,dim=tuple(axes)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            dis=dis/xx
            weight=torch.exp(dis)

            weight[weight_arg==0]=1

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
                # keyP_mask=keyP
                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
                # keyP_mask = keyP_mask[:, 1:]
            sum_gt = (y_onehot*weight).sum(axes) if loss_mask is None else (y_onehot*weight* loss_mask).sum(axes)


        
        
        intersect = (x * y_onehot *weight).sum(axes) if loss_mask is None else (x * y_onehot * weight* loss_mask).sum(axes)
        sum_pred = (x*weight).sum(axes) if loss_mask is None else (x *weight* loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            intersect = AllGatherGrad.apply(intersect).sum(0)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        try:

            dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        except Exception as error:
            print(error)
            print('\n','>>>' * 20)
            print(traceback.print_exc())


        dc=-dc.mean()


        return dc   

class GraphMatchVoxelLoss_RadBranchWeight(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(GraphMatchVoxelLoss_RadBranchWeight, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    
    def soft_dilate(self,img):
        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


    def get_fp_fn(self,x,y,num_classes):
        shp=x.shape
        connection_list=[[2,3,15,16],
                        [1,19,8],
                        [1,20,9],
                        [5,8,11,13],
                        [4],
                        [7,9,12,14],
                        [6],
                        [2,4,19],
                        [3,6,20],
                        [11,12,17,18],
                        [4,10,17],
                        [6,10,18],
                        [4],
                        [6],
                        [1],
                        [1],
                        [10,11],
                        [10,12],
                        [2,8],
                        [3,9]]
        
 
        # shp=vessel.shape
        # adj_matrix=torch.zeros((shp[0], num_classes,num_classes)) # num_classes=21
        fp_mask_all=torch.zeros_like(x,dtype=torch.int32,device=x.device)
        fn_mask_all=torch.zeros_like(x,dtype=torch.int32,device=x.device)
        
        for b in range(shp[0]):
            vessel_=x[b]
            y_vessel=y[b]
            labels_x=torch.unique(vessel_.int())
            labels_y=torch.unique(y_vessel.int())

            fp_mask=torch.zeros_like(vessel_,dtype=torch.int32,device=x.device)
            fn_mask=torch.zeros_like(vessel_,dtype=torch.int32,device=x.device)
            

            for lab in range(num_classes):
                if lab==0:
                    continue

                if (lab not in labels_y) and (lab not in labels_x):
                    continue
                elif lab not in labels_y:
                    # error_count=torch.count_nonzero(vessel_==lab)
                    fp_mask[torch.where(vessel_==lab)]=1
                elif lab not in labels_x:
                    # error_count=torch.count_nonzero(y_vessel==lab)
                    fn_mask[torch.where(y_vessel==lab)]=1
                else:

                    lab_vessel=torch.zeros_like(vessel_,dtype=torch.float32,device=x.device)
                    lab_vessel[torch.where(vessel_==lab)]=1
                    lab_vessel=lab_vessel.unsqueeze(0)

                    dilate_lab_vessel=self.soft_dilate(lab_vessel)
                    dilate_lab_vessel=dilate_lab_vessel.int()
                    dilate_lab_vessel=dilate_lab_vessel.squeeze(0)

                    neigh_vessel=dilate_lab_vessel*vessel_ # 找pred当前类别的邻居
                    neigh_list=torch.unique(neigh_vessel)

                    neigh_y_list=torch.tensor(connection_list[lab-1],device=x.device)
                    x_not_in_y=torch.isin(neigh_list,neigh_y_list,invert=True) #被pred预测错误的邻居
                    y_not_in_x=torch.isin(neigh_y_list,neigh_list,invert=True) #pred中未被预测出的邻居

                    fp_neigh=neigh_list[x_not_in_y]
                    fn_neigh=neigh_y_list[y_not_in_x]

                    if len(fp_neigh)>0:
                        for fp in fp_neigh:
                            # if fp==lab:
                                # print('no.')
                                # continue
                            if fp in labels_y:
                                fp_mask[torch.where((vessel_==lab) & (y_vessel==fp))]=1
                            else:
                                fp_mask[torch.where(vessel_==fp)]=1
                    
                    if len(fn_neigh)>0:
                        for fn in fn_neigh:
                            if fn not in labels_y:
                                continue
                            
                            aa=torch.zeros_like(y_vessel,dtype=torch.bool)
                            aa[torch.where(y_vessel==lab)]=1
                            bb=torch.zeros_like(y_vessel,dtype=torch.bool)
                            bb[torch.where(vessel_==lab)]=1
                            fn_mask[torch.where(torch.logical_xor(aa,bb))]=1
                        
                            aa=torch.zeros_like(y_vessel,dtype=torch.bool)
                            aa[torch.where(y_vessel==fn)]=1
                            bb=torch.zeros_like(y_vessel,dtype=torch.bool)
                            bb[torch.where(vessel_==fn)]=1
                            fn_mask[torch.where(torch.logical_xor(aa,bb))]=1

            fp_mask_all[b]=fp_mask
            fn_mask_all[b]=fn_mask



        return fp_mask_all,fn_mask_all
    

    def forward(self, x, y, weight, loss_mask=None):
        # y=tube skel for label
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            weight=self.apply_nonlin(weight)


        x_class=x.argmax(1)[:, None]

        if not self.do_bg:
            x = x[:, 1:]


        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
                # skel = skel.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))

            weight_arg=weight.argmax(1)[:,None]

            weight=torch.zeros_like(y, device=x.device)

            
            dis=2-(weight_arg-1)/(shp_weight[1]-1)
            xx=torch.amax(dis,dim=tuple(axes)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # print(xx)
            dis=dis/xx
            weight=torch.exp(dis)

            weight[weight_arg==0]=1
                

            # neigh_masks, non_neigh_masks=self.Neighbor_mask(y,x.shape)
            # print('keyP.shape:',keyP.shape)
            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
                # skel_onehot=skel
                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
                # skel_onehot = skel_onehot[:, 1:]

        fp_mask,fn_mask=self.get_fp_fn(x_class,y,21)
        intersect = (x * y_onehot *weight).sum(axes) if loss_mask is None else (x * y_onehot * weight* loss_mask).sum(axes)
        # sum_pred = (x*weight).sum(axes) if loss_mask is None else (x *weight* loss_mask).sum(axes)
        FP_neigh=(x*fp_mask*weight).sum(axes) if loss_mask is None else (x*fp_mask*weight* loss_mask).sum(axes)
        FN_neigh=(x*fn_mask*weight).sum(axes) if loss_mask is None else (x*fn_mask*weight* loss_mask).sum(axes)
    
        
        
        
        if self.ddp and self.batch_dice:
            # mean_mse=AllGatherGrad.apply(mean_mse).sum(0)
            intersect = AllGatherGrad.apply(intersect).sum(0)
            FP_neigh = AllGatherGrad.apply(FP_neigh).sum(0)
            FN_neigh = AllGatherGrad.apply(FN_neigh).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            FP_neigh = FP_neigh.sum(0)
            FN_neigh = FN_neigh.sum(0)


        dc = (2 * intersect + self.smooth) / (torch.clip(2 * intersect +FP_neigh +FN_neigh + self.smooth, 1e-8))
        dc=dc.mean() 


        return -dc


class CubeConnectMSELoss_RadBranchWeight(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, kernel_size: int=3, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(CubeConnectMSELoss_RadBranchWeight, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.kernel_size=kernel_size
        


    def forward(self, x, y, weight, loss_mask=None):
        # y=tube skel for label
        shp_x, shp_y = x.shape, y.shape
        shp_weight=weight.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            weight=self.apply_nonlin(weight)


        x_class=x.argmax(1)[:, None]
        x_onehot = torch.zeros(shp_x, device=x.device)
        x_onehot.scatter_(1, x_class, 1)

        if not self.do_bg:
            x = x[:, 1:]
            x_onehot=x_onehot[:,1:]


        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))


        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
                # skel = skel.view((shp_y[0], 1, *shp_y[1:]))
            if len(shp_x) != len(shp_weight):
                weight = weight.view((shp_weight[0], 1, *shp_weight[1:]))

            weight_arg=weight.argmax(1)[:,None]

            weight=torch.zeros_like(y, device=x.device)

            
            dis=2-(weight_arg-1)/(shp_weight[1]-1)
            xx=torch.amax(dis,dim=tuple(axes)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # print(xx)
            dis=dis/xx
            weight=torch.exp(dis)

            weight[weight_arg==0]=1
                

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
                # skel_onehot=skel
                print('1')
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

                y_vessel=torch.where(y>0,1,0).bool()
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
                # skel_onehot = skel_onehot[:, 1:]
        y_onehot=y_onehot.float()
        kernels=torch.ones((y_onehot.shape[1],1,self.kernel_size,self.kernel_size,self.kernel_size),device=x.device)

        label_connect_onehot=F.conv3d(y_onehot,weight=kernels, padding=self.kernel_size//2,groups=y_onehot.shape[1])
        x_fore_prob=x*x_onehot
        pred_connect_onehot=F.conv3d(x_fore_prob,weight=kernels, padding=self.kernel_size//2,groups=x.shape[1])

        loss_=F.mse_loss(pred_connect_onehot, label_connect_onehot,reduction='none')
        # loss_=F.l1_loss(pred_connect_onehot, label_connect_onehot,reduction='none')
        # print(torch.max(loss_))

        loss_=loss_*weight*y_vessel
        # print(loss_.shape)
        
        loss_=loss_.mean()


        return loss_

class RadiusRegMSELoss(nn.Module):
    def __init__(self):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(RadiusRegMSELoss, self).__init__()


    def forward(self,net_output, target):
        shp_x, shp_y = net_output.shape, target.shape
        net_output=net_output.float()
        target=target.float()
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))
        mse_loss=F.mse_loss(net_output,target).float()

        return mse_loss



       

if __name__ == '__main__':
    import nibabel as nib
    import argparse
    from nnunetv2.utilities.helpers import softmax_helper_dim1



    import nibabel as nib
    import numpy as np

    x_data=nib.load("/data0/user/jlliu/data/nnUNet_process/nnUNet_prediction/External_Validation/nnUNetTrainerWeightRadiuslNoMirroring/predTs_1/ADAM_10003.nii.gz")
    x_img=x_data.get_fdata()
    x_torch=torch.from_numpy(x_img).unsqueeze(0).unsqueeze(0)
    x = x_torch.long()
    x_torch = torch.zeros((1,21,x_torch.shape[2],x_torch.shape[3],x_torch.shape[4]), device=x.device, dtype=torch.float32)
    x_torch.scatter_(1, x, 1)
    print(x_torch.shape)


    y_data=nib.load("/data0/user/jlliu/data/nnUNet_process/nnUNet_prediction/External_Validation/labelsTs/ADAM_10003.nii.gz")
    y_img=y_data.get_fdata()
    y_torch=torch.from_numpy(y_img).unsqueeze(0).unsqueeze(0)
    # y_torch=y_torch.transpose(1,3)
    print(y_torch.shape)

    weight=torch.ones_like(y_torch, dtype=torch.float32)

    # Loss=SoftMulticlDiceLoss(iter_=10,smooth=1e-5)
    # loss_,pred_skel=Loss.forward(y_pred=x_torch,y_true=y_torch)

    # pred=pred_skel.squeeze(0).squeeze(0)
    # pred=pred.numpy()

    # loss_func=MemoryEfficientSoftDiceLoss_RadBranchWeight(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1)
    # loss_func=CubeConnectMSELoss_RadBranchWeight(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1, kernel_size=3)
    # loss_func=Weight_CubeConnectCELoss(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1, kernel_size=3)
    # loss_func=GraphMatchVoxelLoss_RadBranchWeight(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1)
    # loss_func=GraphMatchVoxelLoss_Weight(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1)
    # loss_func=AdjacencyLoss_Weight(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1)
    # loss_func=AdjacencyCoOccurenceLoss_Weight(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1)
    # loss_func=SoftNeighErrorMulticlCELoss(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1)
    loss_func=SoftNeighDirectionalErrorlLoss(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False, apply_nonlin=softmax_helper_dim1)


    # loss_=loss_func(x_torch,y_torch)
    # loss_,label_,pred_=loss_func(x_torch,y_torch)
    loss_,error_map=loss_func(x_torch,y_torch,weight)
    # loss_=loss_func(x_torch,y_torch,weight,weight)
    print(loss_)
    error_map=error_map.squeeze(0)
    error_map=torch.max(error_map,dim=0)[0]
    error_map=error_map.numpy()
    # print(error_map.shape)
    hdr=x_data.header
    hdr['datatype']=16
    nib.save(nib.Nifti1Image(error_map,x_data.affine,hdr),"/data0/user/jlliu/data/nnUNet_process/nnUNet_prediction/External_Validation/ADAM_10003_grad.nii.gz")







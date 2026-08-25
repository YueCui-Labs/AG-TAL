import math
from turtle import up
from SimpleITK import Mask
from numpy import std
import torch
import torch.nn as nn
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.skeletonize_cbdice import Skeletonize
from nnunetv2.training.loss.soft_skeleton_cbdice import SoftSkeletonize
from nnunetv2.utilities.helpers import softmax_helper_dim1

import sys


from PH_losses.betti_losses import FastMulticlassBettiMatchingLoss, FastMulticlass2DBettiMatchingLoss, Multiclass2DWassersteinLoss, Fast2DWassersteinLoss
                

def soft_dice(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    soft_dice: torch.Tensor = 1.0 - coeff
    return soft_dice



#  BettiMatching loss (3D version)
class CE_DC_Betti3D_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,weight_betti=0.5, ignore_label=None,
                 smooth=1.0, num_processes=1,dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(CE_DC_Betti3D_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_betti=weight_betti
        

        self.smooth=smooth
        
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        self.MulticlassBMLoss = FastMulticlassBettiMatchingLoss(
            filtration_type='SUPERLEVEL', 
            num_processes=num_processes,
            convert_to_one_vs_rest=False,
            softmax=True,
            push_unmatched_to_1_0=True,
            ignore_background=True,
            barcode_length_threshold=0.1,
            topology_weights=[0.5,0.5]
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 
        

        if epoch>0:
            bm_loss, _ = self.MulticlassBMLoss(net_output, target_dice)\
                if self.weight_betti != 0 else 0
            result=result+self.weight_betti *bm_loss
        return result

#  BettiMatching loss for multiclass segmentation (2D version, 3D version needs too much training time for each epoch)
class CE_DC_MultiBetti2D_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,weight_betti=0.5, ignore_label=None,
                 smooth=1.0, num_processes=1,dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(CE_DC_MultiBetti2D_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_betti=weight_betti
        

        self.smooth=smooth
        
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        self.MulticlassBMLoss = FastMulticlass2DBettiMatchingLoss(
            filtration_type='SUPERLEVEL', 
            num_processes=num_processes,
            convert_to_one_vs_rest=False,
            softmax=True,
            push_unmatched_to_1_0=True,
            ignore_background=True,
            barcode_length_threshold=0.1,
            topology_weights=[0.5,0.5]
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 
        

        if epoch>0:
            bm_loss, _ = self.MulticlassBMLoss(net_output, target_dice)\
                if self.weight_betti != 0 else 0
            result=result+self.weight_betti *bm_loss

        return result
    

#  HuTopo loss for multiclass segmentation (2D version)
class CE_DC_MultiHutopo2D_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,weight_betti=0.1, ignore_label=None,
                 smooth=1.0, num_processes=1,dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(CE_DC_MultiHutopo2D_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_betti=weight_betti
        

        self.smooth=smooth
        
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        self.MulticlassBMLoss = Multiclass2DWassersteinLoss(
            filtration_type='SUPERLEVEL', 
            num_processes=num_processes,
            convert_to_one_vs_rest=False,
            softmax=True,
            ignore_background=True
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 
        

        if epoch>0:
            bm_loss, _ = self.MulticlassBMLoss(net_output, target_dice)\
                if self.weight_betti != 0 else 0
            result=result+self.weight_betti *bm_loss



        return result
    
#  HuTopo loss for binary segmentation (2D version)
class CE_DC_Hutopo2D_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,weight_betti=0.1, ignore_label=None,
                 smooth=1.0, num_processes=1,dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(CE_DC_Hutopo2D_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_betti=weight_betti
        

        self.smooth=smooth
        
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        self.MulticlassBMLoss = Fast2DWassersteinLoss(
            filtration_type='SUPERLEVEL', 
            num_processes=num_processes,
            convert_to_one_vs_rest=False,
            softmax=True,
            ignore_background=True
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 
        

        if epoch>0:
            bm_loss, _ = self.MulticlassBMLoss(net_output, target_dice)\
                if self.weight_betti != 0 else 0
            result=result+self.weight_betti *bm_loss


        return result



if __name__ == '__main__':
    import nibabel as nib
    import argparse
    from nnunetv2.utilities.helpers import softmax_helper_dim1


    x=torch.randn(2,21,64,160,160)
    y=torch.ones(2,1,64,160,160)
    y[:,:,7,:,:]=6
    y[:,:,20,:,:]=20
    y[:,:,13,:,:]=14
    y[:,:,1,:,:]=0
    weight=torch.ones(2,1,64,160,160)
    weight[:,:,7,:,:]=2
    weight[:,:,20,:,:]=2
    weight[:,:,13,:,:]=2
    weight[:,:,1,:,:]=2

    x_lab=torch.argmax(x,dim=1,keepdim=True)
    # print(torch.unique(x_lab))


    loss_func=CE_DC_Hutopo2D_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {},
                                    weight_ce=1, weight_dice=1, weight_betti=0.1,num_processes=1,smooth=1e-5, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
    loss_=loss_func(x,y,1)
    print('loss:',loss_)
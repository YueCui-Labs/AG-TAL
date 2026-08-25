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

             

def soft_dice(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    soft_dice: torch.Tensor = 1.0 - coeff
    return soft_dice


# Dice+clDice loss
class dice_cldice_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_cldice=1):
        super(dice_cldice_loss, self).__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_dice=weight_dice
        self.weight_cldice=weight_cldice

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:#def forward(y_true, y_pred):
        y_pred=y_pred.softmax(dim=1)

        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        dice = soft_dice(y_true_oh, y_pred, self.smooth)

        skel_pred = soft_skel(y_pred, self.iter_)
        skel_true = soft_skel(y_true_oh, self.iter_)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true_oh)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.0- 2.0*(tprec*tsens)/(tprec+tsens)
        #total_loss: torch.Tensor = (1.0 - self.alpha) * dice + self.alpha * cl_dice
        result = self.weight_dice * dice + self.weight_cldice * cl_dice
        return result

# Dice+clCE loss
class dice_clCE_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1):
        super(dice_clCE_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.weight_clCE = weight_clCE
        self.weight_dice = weight_dice

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)

        dice = soft_dice(y_true_oh, y_pred, self.smooth)

        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = self.weight_dice * dice + self.weight_clCE * cl_ce
        return result



#CE + clDice loss
class CE_cldice_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=3, smooth=1.0, weight_ce=1, weight_cldice=1, ignore_label=None):
        super(CE_cldice_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_cldice = weight_cldice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None) else 0

        target_oh = torch.zeros(net_output.shape, device=net_output.device)
        target_oh.scatter_(1, target.long(), 1) # nnUNet-training-loss-dice.py

        net_output=net_output.softmax(dim=1)
        skel_true = soft_skel(target_oh, self.iter_)
        skel_pred = soft_skel(net_output, self.iter_)
        #tprec = (torch.sum(torch.multiply(skel_pred, target[:, 0])[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tprec = (torch.sum(torch.multiply(skel_pred, target_oh)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, net_output)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.0- 2.0*(tprec*tsens)/(tprec+tsens)

        ##Total loss computation##
        result = self.weight_ce * ce_loss + self.weight_cldice * cl_dice
        return result

# CE+clCE loss
class CE_clCE_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=3, weight_ce=1, weight_clCE=1):
        super(CE_clCE_loss, self).__init__()
        self.iter = iter_
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.weight_clCE = weight_clCE
        self.weight_ce = weight_ce

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        ce_loss = self.ce(y_pred, y_true[:, 0])
        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = self.weight_ce * ce_loss + self.weight_clCE * cl_ce
        return result
    

# Dice+CE+clCE loss
class CE_DC_clCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,weight_clCE=1, ignore_label=None,
                 smooth=1.0, iter_=3,dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(CE_DC_clCE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_clce=weight_clCE
        
        self.iter = iter_
        self.smooth=smooth
        
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.clCE=SoftclCELoss(iter_=self.iter,smooth=self.smooth)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
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
        clce_loss=self.clCE(net_output,target)\
            if self.weight_clce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss +self.weight_clce *clce_loss
        return result

# clCE loss for multiclass segmentation
class SoftclCELoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftclCELoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):

        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true[:,0].long(), reduction="none")
        
        
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]

        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)

        skel_pred_prob = skel_pred_hard * y_pred_prob

        tprec = torch.mul(cross_ent, skel_true)

        tprec=tprec.nanmean()
        tsens = torch.mul(cross_ent, skel_pred_prob)

        tsens=tsens.nanmean()
        cl_ce = (tprec+tsens)

        return cl_ce


# clCE loss for directly multiclass segmentation (needs a lot of CUDA memory, abandoned)
class SoftMulticlCELoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftMulticlCELoss, self).__init__()
        self.smooth = smooth
        self.ce = RobustCrossEntropyLoss(reduction='none',label_smoothing=smooth)
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):

        exp_y_pred=torch.exp(y_pred)
        log_smooth=math.log(self.smooth)
        if (exp_y_pred<self.smooth).any():
            print('need modify!')
            print('torch.min(y_pred):',torch.min(y_pred))
            y_pred=torch.where(exp_y_pred<self.smooth,log_smooth,y_pred)

        
        
        cross_ent = self.ce(y_pred, y_true[:,0].long())
        cross_ent=cross_ent.unsqueeze(1)
        print('cross_ent.shape:',cross_ent.shape)


        if torch.isnan(cross_ent).int().sum()>0:
            print('there is nan in cross_ent!')
            print('the number of nan:',torch.isnan(cross_ent).int().sum())
        if torch.isinf(cross_ent).int().sum()>0:
            print('there is inf in cross_ent!')
            print('the number of inf:',torch.isinf(cross_ent).int().sum())

            map_y_pred=y_pred[torch.where(torch.isinf(cross_ent))]

            map_y_true=y_true[torch.where(torch.isinf(cross_ent))]
            print('map_y_pred:',map_y_pred)
            print('map_y_true:',map_y_true)
        if torch.count_nonzero(y_pred)==0:
            print('there is not non-zero in y_pred!')
        if torch.count_nonzero(y_true[:,0])==0:
            print('there is not non-zero in y_true!')    

        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py
        y_true_oh=y_true_oh[:,1:]
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]

        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:

                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1))
                skel_true = self.m_skeletonize(y_true.unsqueeze(1))

        skel_pred_prob = skel_pred_hard * y_pred[:, 1:]
        skel_true_oh = skel_true * y_true_oh
        print('skel_true_oh.shape:',skel_true_oh.shape)
        print('skel_pred_prob.shape:',skel_pred_prob.shape)

        if torch.isnan(skel_pred_prob).int().sum()>0:
            print('there is nan in skel_pred_prob!')
            print('the number of nan:',torch.isnan(skel_pred_prob).int().sum())

        if torch.isnan(skel_true_oh).int().sum()>0:
            print('there is nan in skel_true_oh!')
            print('the number of nan:',torch.isnan(skel_true_oh).int().sum())

        if torch.isinf(skel_pred_prob).int().sum()>0:
            print('there is inf in skel_pred_prob!')
            print('the number of inf:',torch.isinf(skel_pred_prob).int().sum())

        if torch.isinf(skel_true_oh).int().sum()>0:
            print('there is inf in skel_true_oh!')
            print('the number of inf:',torch.isinf(skel_true_oh).int().sum())

        
        tprec = torch.mul(cross_ent, skel_true_oh)
        if torch.isnan(tprec).int().sum()>0:
            print('there is nan in tprec!')
            print('the number of nan:',torch.isnan(tprec).int().sum())

        if torch.isinf(tprec).int().sum()>0:
            print('there is inf in tprec!')
            print('the number of inf:',torch.isinf(tprec).int().sum())
        tprec=tprec.nanmean()
        tsens = torch.mul(cross_ent, skel_pred_prob)
        if torch.isnan(tsens).int().sum()>0:
            print('there is nan in tsens!')
            print('the number of nan:',torch.isnan(tsens).int().sum())

        if torch.isinf(tsens).int().sum()>0:
            print('there is inf in tsens!')
            print('the number of inf:',torch.isinf(tsens).int().sum())
        tsens=tsens.nanmean()
        print("tprec:",tprec)
        print("tsens:",tsens)
        cl_ce = (tprec+tsens)


        return cl_ce
    



if __name__ == '__main__':
    import nibabel as nib
    import argparse
    from nnunetv2.utilities.helpers import softmax_helper_dim1


    x=torch.randn(2,21,64,160,160)
    # x=torch.zeros(2,21,64,160,160)
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

    loss_func=SoftMulticlCELoss(iter_=10,smooth=1e-3)
    # 
    loss_=loss_func(x,y)
    print('loss:',loss_)


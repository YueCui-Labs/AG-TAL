import torch
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

import torch
from nnunetv2.training.loss.skeletonize_cbdice import Skeletonize
from nnunetv2.training.loss.soft_skeleton_cbdice import SoftSkeletonize

# clDice loss for binary segmentation
class SoftclDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftclDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        
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

        tprec = (torch.sum(torch.multiply(skel_pred_prob, y_true))+self.smooth)/(torch.sum(skel_pred_prob)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred_prob))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice_loss = - 2.0 * (tprec*tsens)/(tprec+tsens)

        return cl_dice_loss
    

# clDice loss for multiclass segmentation (revised, do not need much CUDA memory)
class SoftMulticlDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftMulticlDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        
        # print(torch.unique(y_true))
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
                # skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                # skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1))
                skel_true = self.m_skeletonize(y_true.unsqueeze(1))

        # skel_pred_prob = skel_pred_hard * y_pred_prob
        skel_pred_prob = skel_pred_hard * y_pred[:, 1:]
        skel_true_oh = skel_true * y_true_oh


        tprec = (torch.sum(torch.multiply(skel_pred_prob, y_true_oh))+self.smooth)/(torch.sum(skel_pred_prob)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true_oh, y_pred[:, 1:]))+self.smooth)/(torch.sum(skel_true_oh)+self.smooth)
        cl_dice_loss = - 2.0 * (tprec*tsens)/(tprec+tsens)
        cl_dice_loss=cl_dice_loss.mean()
        

        return cl_dice_loss
    

class DualSoftMulticlDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.,weight_1to2=0.5,weight_2to1=0.5):
        super(DualSoftMulticlDiceLoss, self).__init__()
        self.smooth = smooth
        self.weight_1_label=weight_1to2
        self.weight_2_label=weight_2to1
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_main, y_radius, t_skeletonize_flage=False):
        cl_1_label=self.forward_one(y_pred=y_radius,y_true=y_main,t_skeletonize_flage=t_skeletonize_flage)
        cl_2_label=self.forward_one(y_pred=y_main,y_true=y_radius,t_skeletonize_flage=t_skeletonize_flage)

        cldc=self.weight_1_label*cl_1_label+self.weight_2_label*cl_2_label
        return cldc

    def forward_one(self, y_pred, y_true, t_skeletonize_flage=False):
        
        shp_pred,shp_true=y_pred.shape,y_true.shape
        if shp_pred[1]==shp_true[1]:
            y_true_softmax=torch.softmax(y_true, 1)
            y_true=y_true_softmax.argmax(1)[:,None]
            shp_true=y_true.shape
        if len(shp_pred) != len(shp_true):
            y_true = y_true.view((shp_true[0], 1, *shp_true[1:]))
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
                # skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                # skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1))
                skel_true = self.m_skeletonize(y_true.unsqueeze(1))

        # skel_pred_prob = skel_pred_hard * y_pred_prob
        skel_pred_prob = skel_pred_hard * y_pred[:, 1:]
        skel_true_oh = skel_true * y_true_oh


        tprec = (torch.sum(torch.multiply(skel_pred_prob, y_true_oh))+self.smooth)/(torch.sum(skel_pred_prob)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true_oh, y_pred[:, 1:]))+self.smooth)/(torch.sum(skel_true_oh)+self.smooth)
        cl_dice_loss = - 2.0 * (tprec*tsens)/(tprec+tsens)
        # cl_dice_loss=cl_dice_loss.mean()

        return cl_dice_loss
    

class SoftConsistentclDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftConsistentclDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

        

    def forward(self, y_pred, y_radius, t_skeletonize_flage=False):
        

        # y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        # y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py
        # y_true_oh=y_true_oh[:,1:]
        # y_pred_=torch.softmax(y_pred, 1)
        # y_pred_=y_pred_.argmax(1)[:, None]
        # y_pred_onehot = torch.zeros_like(y_pred, device=x.device)
        # y_pred_onehot

        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]

        y_radius_fore = y_radius[:, 1:]
        y_radius_fore = torch.max(y_radius_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_radius_binary = torch.cat([y_radius[:, :1], y_radius_fore], dim=1)
        y_radius_binary = torch.softmax(y_radius_binary, 1)
        y_radius_prob = y_radius_binary[:, 1]

        with torch.no_grad():
            # y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
            y_radius_hard = (y_radius_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_radius_hard = self.t_skeletonize(y_radius_hard.unsqueeze(1)).squeeze(1)
            else:
                # skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                # skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1))
                skel_radius_hard = self.m_skeletonize(y_radius_hard.unsqueeze(1))

        # skel_pred_prob = skel_pred_hard * y_pred_prob
        # skel_pred_prob = skel_pred_hard * y_pred[:, 1:]
        # skel_radius_prob = skel_radius_hard * y_radius[:, 1:]
        skel_pred_prob = skel_pred_hard * y_pred_fore
        # print('skel_pred_prob.shape:',skel_pred_prob.shape)
        skel_radius_prob = skel_radius_hard * y_radius_fore
        # print('skel_radius_prob.shape:',skel_radius_prob.shape)
        # skel_true_oh = skel_true * y_true_oh
        y_radius_oh=y_radius_hard.unsqueeze(1)
        # print('y_radius_oh.shape:',y_radius_oh.shape)
        y_pred_oh=y_pred_hard.unsqueeze(1)
        # print('y_pred_oh.shape:',y_pred_oh.shape)
        


        tprec = (torch.sum(torch.multiply(skel_pred_prob, y_radius_oh))+self.smooth)/(torch.sum(skel_pred_prob)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_radius_prob, y_pred_oh))+self.smooth)/(torch.sum(skel_radius_prob)+self.smooth)
        cl_dice_loss = - 2.0 * (tprec*tsens)/(tprec+tsens)
        # cl_dice_loss=cl_dice_loss.mean()

        return cl_dice_loss

# Dice+CE+clDice(for binary) loss
class DC_and_CE_and_CLDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cldc_kwargs, weight_ce=1, weight_dice=1, weight_cldice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_CLDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cldice = SoftclDiceLoss(**cldc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t_skeletonize_flage=False):
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

        cldice_loss = self.cldice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_cldice != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldice_loss
        return result


# Dice+CE+clDice(for multiclass) loss
class DC_and_CE_and_MultiCLDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cldc_kwargs, weight_ce=1, weight_dice=1, weight_cldice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_MultiCLDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cldice = SoftMulticlDiceLoss(**cldc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t_skeletonize_flage=False):
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

        cldice_loss = self.cldice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_cldice != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldice_loss
        return result



if __name__ == '__main__':


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
    print(y_torch.shape)

    Loss=SoftMulticlDiceLoss(iter_=10,smooth=1e-5)
    loss_,pred_skel=Loss.forward(y_pred=x_torch,y_true=y_torch)

    pred=pred_skel.squeeze(0).squeeze(0)
    pred=pred.numpy()

    nib.save(nib.Nifti1Image(pred,y_data.affine,y_data.header),"/data0/user/jlliu/data/nnUNet_process/nnUNet_prediction/External_Validation/ADAM_10003_clskel.nii.gz")

    print(pred_skel.shape)
    print('loss:',loss_)









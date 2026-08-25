from nnunetv2.training.loss.compound_cldice_loss import DualSoftMulticlDiceLoss, SoftConsistentclDiceLoss
import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.custom_Loss import AdjacencyCoOccurenceLoss_Weight, AdjacencyLoss_Weight, CubeConnectMSELoss_RadBranchWeight, GraphMatchVoxelLoss_RadBranchWeight, MemoryEfficientSoftDiceLoss_RadBranchWeight, RadiusRegMSELoss, SoftNeighErrorMultiSkelRecallLoss, SoftNeighErrorMulticlCELoss, SoftNeighErrorMulticlDiceLoss
from nnunetv2.training.loss.custom_Loss import MemoryEfficientSoftDiceLoss_Weight
from nnunetv2.training.loss.custom_Loss import GraphMatchVoxelLoss_Weight,DPCrossEntropyLoss,DPFocalLoss,FocalLoss3D
from nnunetv2.training.loss.custom_Loss import Weight_CrossEntropyLoss, Weight_CubeConnectCELoss

from torch import nn
from .soft_skeleton import soft_skel


    
# baseline loss(dice+CE) + radius classification loss
class DC_and_CE_RadCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_rad=0.5, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_RadCE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_rad = weight_rad
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        # self.radCE = DPCrossEntropyLoss()
        # self.radCE = DPFocalLoss()
        self.radCE = FocalLoss3D(alpha = torch.tensor([0.02, 0.3, 0.3, 0.2, 0.1, 0.08]))

    def forward(self, net_output_main: torch.Tensor,net_output_aux: torch.Tensor, target: torch.Tensor, radius: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_radius = torch.where(mask, radius, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_radius = radius
            mask = None

        dc_loss = self.dc(net_output_main, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        radCE_loss = self.radCE(net_output_aux, target_radius) \
            if self.weight_rad != 0 and (self.ignore_label is None or num_fg > 0) else 0
        ce_loss = self.ce(net_output_main, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_rad*radCE_loss
        return result

 
# AG-TAL without breakage-aware MSE loss (ablation study of MICCAI version)
class DC_GMVoxel_Weight_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,
                 ignore_label=None, do_GMatching=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_GMVoxel_Weight_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=GraphMatchVoxelLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            mask = None

        
        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss
        return result


# AG-TAL without adjacency-aware Dice loss (ablation study of MICCAI version)
class DC_CCMSE_Weight_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_cc=1,kernel_size=3,
                 ignore_label=None, do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_CCMSE_Weight_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cc = weight_cc
        self.ignore_label = ignore_label
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=Weight_CubeConnectCELoss(apply_nonlin=softmax_helper_dim1, kernel_size=kernel_size, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            mask = None

        
        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
       

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_cc  != 0 else 0
            
                result = result +self.weight_cc *neigh_loss
        return result


# AG-TAL (MICCAI version)
class DCWeight_CE_GMVoxel_NeighMSE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_GMVoxel_NeighMSE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=GraphMatchVoxelLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=Weight_CubeConnectCELoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            mask = None

        
        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result
    


# radius-aware Dice loss + breakage-aware MSE loss +  the differentiable version of adjacency-aware dice loss (after rebuttal)
class DCWeight_CE_AdjV2_NeighMSE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV2_NeighMSE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=Weight_CubeConnectCELoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight = keyP_weight
            mask = None

        # if self.do_wass:
        #     if epoch<30:
        #         dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
        #             if self.weight_dice != 0 else 0
        #     else:
        #         dc_loss = self.dc_wass(net_output, target_dice, target_weight, loss_mask=mask) \
        #             if self.weight_dice != 0 else 0
        # else:
        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        # g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
        #         if self.weight_gm != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>40:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result

# radius-aware Dice loss + breakage-aware clDice loss + the differentiable version of adjacency-aware dice loss (after rebuttal)
class DCWeight_CE_AdjV2_NeighclDice_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV2_NeighclDice_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=SoftNeighErrorMulticlDiceLoss(apply_nonlin=softmax_helper_dim1,iter_=10, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight = keyP_weight
            mask = None

        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0


        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,weight=target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result
    
# radius-aware Dice loss + breakage-aware clCE loss + the differentiable version of adjacency-aware dice loss (after rebuttal)
class DCWeight_CE_AdjV2_NeighclCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV2_NeighclCE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=SoftNeighErrorMulticlCELoss(apply_nonlin=softmax_helper_dim1,iter_=10, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight = keyP_weight
            mask = None

        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        # g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
        #         if self.weight_gm != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,weight=target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result

# radius-aware Dice loss + breakage-aware SkelRecall loss + the differentiable version of adjacency-aware dice loss (after rebuttal)
class DCWeight_CE_AdjV2_NeighSkelRecall_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV2_NeighSkelRecall_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=SoftNeighErrorMultiSkelRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
            # self.neigh_loss=Weight_CubeConnectCELoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor,target_skel: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, target_skel, 0)
            target_weight = torch.where(mask, weight, 0)
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight=keyP_weight

            mask = None

        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_skel,target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result


# radius-aware Dice loss + breakage-aware MSE loss + adjacency-aware co-occurrence loss
class DCWeight_CE_AdjV3_NeighMSE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV3_NeighMSE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyCoOccurenceLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=Weight_CubeConnectCELoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight = keyP_weight
            mask = None

        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0


        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result
    

# AG-TAL (journal version): radius-aware Dice loss + breakage-aware clDice loss + adjacency-aware co-occurrence loss
class DCWeight_CE_AdjV3_NeighclDice_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV3_NeighclDice_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyCoOccurenceLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=SoftNeighErrorMulticlDiceLoss(apply_nonlin=softmax_helper_dim1,iter_=10, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight = keyP_weight
            mask = None

        # if self.do_wass:
        #     if epoch<30:
        #         dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
        #             if self.weight_dice != 0 else 0
        #     else:
        #         dc_loss = self.dc_wass(net_output, target_dice, target_weight, loss_mask=mask) \
        #             if self.weight_dice != 0 else 0
        # else:
        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        # g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
        #         if self.weight_gm != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,weight=target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result
    
# radius-aware Dice loss + breakage-aware clCE loss + adjacency-aware co-occurrence loss
class DCWeight_CE_AdjV3_NeighclCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV3_NeighclCE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyCoOccurenceLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=SoftNeighErrorMulticlCELoss(apply_nonlin=softmax_helper_dim1,iter_=10, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight = keyP_weight
            mask = None

        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        # g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
        #         if self.weight_gm != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,weight=target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result

# AG-TAL (journal version) + radius classification loss
class DCWeight_CE_AdjV3_NeighclDice_RadCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,weight_rad=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV3_NeighclDice_RadCE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        self.weight_rad=weight_rad
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        self.radCE = FocalLoss3D(alpha = torch.tensor([0.02, 0.3, 0.3, 0.2, 0.1, 0.08]))

        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyCoOccurenceLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=SoftNeighErrorMulticlDiceLoss(apply_nonlin=softmax_helper_dim1,iter_=10, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor,net_output_aux: torch.Tensor, target: torch.Tensor,target_aux: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            target_aux = torch.where(mask, target_aux, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight = keyP_weight
            target_aux = target_aux
            mask = None
            num_fg = torch.tensor(1, device=net_output.device)


        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) 
        ce_loss = self.ce(net_output, target[:, 0])
        radCE_loss = self.radCE(net_output_aux, target_aux)
        

        if self.do_GraphMatching:
            gm_gate = self.weight_gm*float(epoch > 30)
            g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask)
        else:
            g_matching_loss=torch.zeros((), device=net_output.device)
            gm_gate=0.0

        if self.do_NeighMSE:
            neigh_gate = self.weight_neigh*float(epoch > 30)
            neigh_loss=self.neigh_loss(net_output, target_dice,weight=target_weight, loss_mask=mask)
        else:
            neigh_loss=torch.zeros((), device=net_output.device)
            neigh_gate=0.0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss +self.weight_rad*radCE_loss + gm_gate*g_matching_loss + neigh_gate*neigh_loss
        return result


# radius-aware Dice loss + breakage-aware SkelRecall loss + adjacency-aware co-occurrence loss
class DCWeight_CE_AdjV3_NeighSkelRecall_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_AdjV3_NeighSkelRecall_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=AdjacencyCoOccurenceLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=SoftNeighErrorMultiSkelRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)




    def forward(self, net_output: torch.Tensor, target: torch.Tensor,target_skel: torch.Tensor, weight: torch.Tensor,keyP_weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, target_skel, 0)
            target_weight = torch.where(mask, weight, 0)
            target_keyP_weight = torch.where(mask, keyP_weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_keyP_weight=keyP_weight
            # target_skel=target_skel
            mask = None


        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 

        if self.do_GraphMatching:
            if epoch>30:
                # g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight,target_keyP_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_skel,target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result
    

# radius-aware CE loss (abandoned)
class Weight_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, 
                 ignore_label=None, do_Wass=False, do_GMatching=False, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Weight_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

        self.ignore_label = ignore_label
        self.do_wass=do_Wass
        self.do_GraphMatching=do_GMatching

        self.ce = Weight_CrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            mask = None

        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        ce_loss = self.ce(net_output, target[:, 0], target_weight) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0


        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 
        return result


class DC_CE_GMVoxel_NeighMSE_RadBranchWeightloss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,weight_rad=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_RadBranchWeight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_CE_GMVoxel_NeighMSE_RadBranchWeightloss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        self.weight_rad=weight_rad
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=GraphMatchVoxelLoss_RadBranchWeight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=CubeConnectMSELoss_RadBranchWeight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)


        self.radCE = FocalLoss3D(alpha = torch.tensor([0.02, 0.3, 0.3, 0.2, 0.1, 0.08]))



    def forward(self, net_output: torch.Tensor,net_output_aux: torch.Tensor, target: torch.Tensor, radius: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_radius = torch.where(mask, radius, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_radius = radius
            mask = None


        dc_loss = self.dc(net_output, target_dice, net_output_aux, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        radCE_loss = self.radCE(net_output_aux, target_radius) \
            if self.weight_rad != 0 and (self.ignore_label is None or num_fg > 0) else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss +self.weight_rad*radCE_loss

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,net_output_aux, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,net_output_aux, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result


# AG-TAL (MICCAI version) + radius classification loss
class DCWeight_CE_GMVoxel_NeighMSE_RadCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_gm=1,weight_neigh=1,weight_rad=1,
                 ignore_label=None, do_GMatching=True,do_NeighMSE=True, dice_class=MemoryEfficientSoftDiceLoss_Weight):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DCWeight_CE_GMVoxel_NeighMSE_RadCE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_gm = weight_gm
        self.ignore_label = ignore_label
        self.weight_neigh = weight_neigh
        self.weight_rad=weight_rad
        

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.radCE = FocalLoss3D(alpha = torch.tensor([0.02, 0.3, 0.3, 0.2, 0.1, 0.08]))

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_GraphMatching=do_GMatching
        if self.do_GraphMatching:
            self.g_matching_loss=GraphMatchVoxelLoss_Weight(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.do_NeighMSE=do_NeighMSE
        if self.do_NeighMSE:
            self.neigh_loss=Weight_CubeConnectCELoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor,net_output_aux: torch.Tensor, target: torch.Tensor,target_aux: torch.Tensor, weight: torch.Tensor, epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_weight = torch.where(mask, weight, 0)
            target_aux = torch.where(mask, target_aux, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_weight = weight
            target_aux = target_aux
            mask = None

        dc_loss = self.dc(net_output, target_dice, target_weight, loss_mask=mask) \
                if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        radCE_loss = self.radCE(net_output_aux, target_aux) \
            if self.weight_rad != 0 and (self.ignore_label is None or num_fg > 0) else 0
        


        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss +self.weight_rad*radCE_loss

        if self.do_GraphMatching:
            if epoch>30:
                g_matching_loss=self.g_matching_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_gm != 0 else 0
            
                result = result +self.weight_gm*g_matching_loss

        if self.do_NeighMSE:
            if epoch>30:
                neigh_loss=self.neigh_loss(net_output, target_dice,target_weight, loss_mask=mask) \
                if self.weight_neigh  != 0 else 0
            
                result = result +self.weight_neigh *neigh_loss
        return result
    


    
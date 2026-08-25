from nnunetv2.configuration import get_allowed_n_proc_DA
from nnunetv2.training.loss.compound_clCE_loss import dice_cldice_loss, CE_cldice_loss, CE_clCE_loss, dice_clCE_loss, CE_DC_clCE_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans_variants
import torch
import numpy as np
import warnings
from torch import autocast
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision

# dice+cldice
class nnUNetTrainerDiceclDiceLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = dice_cldice_loss(iter_=3, smooth=1.0, weight_dice=1, weight_cldice=1)
        return loss
# dice+clCE
class nnUNetTrainerDiceclCELoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = dice_clCE_loss(iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1)
        return loss


# CE+cldice
class nnUNetTrainerCEclDiceLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = CE_cldice_loss({}, iter_=3, smooth=1.0, weight_ce=1, weight_cldice=1,
                              ignore_label=self.label_manager.ignore_label)
        return loss

# CE+clCE
class nnUNetTrainerCEclCEloss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.enable_deep_supervision = False
        self.num_epochs = 300
    def _build_loss(self):
        loss = CE_clCE_loss({}, iter_=3, weight_ce=1, weight_clCE=1)
        return loss
# Dice+CE+clCE+nomirroring
class nnUNetTrainerCEclCEDiceloss_NoMirroring(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # self.weight_srec = 1 # This is the default value, you can change it if you want
        self.num_val_iterations_per_epoch = 50
        # self.num_epochs = 100
        self.num_epochs = 200 #改成160个epoch，短期测试一下
        print('self.num_epochs:',self.num_epochs)
        if self.label_manager.has_regions:
            raise NotImplementedError("trainer not implemented for regions")
        # self.enable_deep_supervision = False
    
    # def _build_loss(self):
    #     loss=CE_DC_clCE_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
    #                                 weight_ce=1, weight_dice=1, weight_clCE=1,iter_=10,smooth=1e-5, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

    #     return loss
    def _build_loss(self):
        if self.label_manager.ignore_label is not None:
            warnings.warn('Support for ignore label with Skeleton Recall is experimental and may not work as expected')
        loss = CE_DC_clCE_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    weight_ce=1, weight_dice=1, weight_clCE=1,iter_=10,smooth=1e-5, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # if self.enable_deep_supervision:
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

# Dice+CE+clCE+resUNet+nomirroring,对比实验
class nnUNetTrainerCEclCEDiceloss_ResEncoder_NoMirroring(nnUNetTrainerCEclCEDiceloss_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # self.weight_srec = 1 # This is the default value, you can change it if you want
        self.num_val_iterations_per_epoch = 50
        # self.num_epochs = 100
        self.num_epochs = 200 #改成160个epoch，短期测试一下
        print('self.num_epochs:',self.num_epochs)
        if self.label_manager.has_regions:
            raise NotImplementedError("trainer not implemented for regions")
        
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True):
        """
        his is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        return get_network_from_plans_variants('ResidualEncoderUNet',plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)


    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # self.weight_srec = 1 # This is the default value, you can change it if you want
        self.num_val_iterations_per_epoch = 50
        # self.num_epochs = 100
        self.num_epochs = 200 #改成160个epoch，短期测试一下
        print('self.num_epochs:',self.num_epochs)
        if self.label_manager.has_regions:
            raise NotImplementedError("trainer not implemented for regions")


    def _build_loss(self):
        if self.label_manager.ignore_label is not None:
            warnings.warn('Support for ignore label with Skeleton Recall is experimental and may not work as expected')
        loss = CE_DC_Hutopo2D_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    weight_ce=1, weight_dice=1, weight_betti=0.1,num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),smooth=1e-5, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # if self.enable_deep_supervision:
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
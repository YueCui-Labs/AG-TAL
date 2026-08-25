from nnunetv2.configuration import get_allowed_n_proc_DA
from nnunetv2.training.loss.compound_BettiMatching_loss import CE_DC_Betti3D_loss, CE_DC_Hutopo2D_loss, CE_DC_MultiBetti2D_loss, CE_DC_MultiHutopo2D_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans_variants
import torch
import numpy as np
import warnings
from torch import autocast
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision


# BettiMatching 3D模式（训练时间长）
class nnUNetTrainer_ResEncoder_CE_DC_Betti3D_NoMirroring(nnUNetTrainer):
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
        loss = CE_DC_Betti3D_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    weight_ce=1, weight_dice=1, weight_betti=0.05,num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),smooth=1e-5, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

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


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        # weight = batch['weight']

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0].cpu().numpy(), name='data')
        # viewer.add_image(target[0][0].cpu().numpy(), name='target')
        # viewer.add_image(skel[0][0].cpu().numpy(), name='skel')
        # napari.run()

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            # weight = [i.to(self.device, non_blocking=True) for i in weight]
            # for i in range(len(weight)):
            #     print('weight.shape:',weight[i].shape)
            #     print('target.shape:',target[i].shape)
        else:
            target = target.to(self.device, non_blocking=True)
            # weight = weight.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            current_epoch_list=[self.current_epoch for i in range(len(target))]
            l = self.loss(output, target, current_epoch_list)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        # weight = batch['weight']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            # weight = [i.to(self.device, non_blocking=True) for i in weight]
        else:
            target = target.to(self.device, non_blocking=True)
            # weight = weight.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            current_epoch_list=[self.current_epoch for i in range(len(target))]
            
            l = self.loss(output, target, current_epoch_list)

        # we only need the output with the highest output resolution (if DS enabled)
        # if self.enable_deep_supervision:
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    

# MulticlassBettiMatching 2D模式，对比实验
class nnUNetTrainer_ResEncoder_CE_DC_MultiBetti2D_NoMirroring(nnUNetTrainer_ResEncoder_CE_DC_Betti3D_NoMirroring):
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
        loss = CE_DC_MultiBetti2D_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    weight_ce=1, weight_dice=1, weight_betti=0.05,num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),smooth=1e-5, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

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
    
# MulticlassHuTopo 2D模式，对比实验
class nnUNetTrainer_ResEncoder_CE_DC_MultiHuTopo2D_NoMirroring(nnUNetTrainer_ResEncoder_CE_DC_Betti3D_NoMirroring):
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
        loss = CE_DC_MultiHutopo2D_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
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

# HuTopo 2D模式
class nnUNetTrainer_ResEncoder_CE_DC_HuTopo2D_NoMirroring(nnUNetTrainer_ResEncoder_CE_DC_Betti3D_NoMirroring):
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
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.network_trainer import NetworkTrainer
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.drloc_loss import cal_selfsupervised_loss
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.network_architecture.models.nnFormer import nnformer
from nnunet.network_architecture.models.nnFormer_75m import nnformer_75m
from nnunet.network_architecture.models.nnFormer_300m import nnformer_300m
from nnunet.network_architecture.models.nnFormer_pool import nnformer_pool
from nnunet.network_architecture.models.nnFormer_pool_MAE import nnformer_pool_mae
from nnunet.network_architecture.models.nnFormer_LNOff import nnformer_LNOff
from nnunet.network_architecture.models.nnFormer_absolute import nnformer_absolute
from nnunet.network_architecture.models.nnFormer_noPos import nnformer_noPos
from nnunet.network_architecture.models.nnFormer_absolute_MAE import nnformer_absolute_mae
from nnunet.network_architecture.models.nnFormer_noPos_MAE import nnformer_noPos_mae
from nnunet.network_architecture.models.nnConv import nnConv
from nnunet.network_architecture.models.nnFormer_MAE import nnformer_mae
from nnunet.network_architecture.models.nnConv_MAE import nnConv_mae
from nnunet.network_architecture.models.nnFormer_sinusoid import nnformer_sinusoid
from nnunet.network_architecture.models.nnFormer_sinusoid_MAE import nnformer_sinusoid_mae
from nnunet.network_architecture.models.nnFormer_auxiliary import nnformer_auxiliary
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
import numpy as np
import torch
from collections import OrderedDict
from typing import Tuple

#model_name = '3d_nnFormer'
#splits_path = 'splits_h2.pkl'
class TransformerUNetTrainer(nnUNetTrainer):
    '''
    Modified trainer for Transformer
    The run_training.py first gets the trainer class then initializes it
    trainer = trainer_class()
    trainer.initialize()
    optional(load check_points)
    trainer.run_training()

    trainer.load...
    trainer.validate()
    '''
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, splits_path=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, sampling_method=None):
        '''

        :param plans_file: '/home/yicong/桌面/nnUNet_Folders/nnUNet_preprocessed/Task501_ProstateSegmentation/nnUNetPlansv2.1_plans_3D.pkl' won't change always use this one
        :param fold: 0 won't change always use this one
        :param output_folder: '/home/yicong/桌面/nnUNet_Folders/RESULTS_FOLDER/nnUNet/3d_nnFormer/Task501_ProstateSegmentation/TransformerUNetTrainer__nnUNetPlansv2.1' model based
        :param dataset_directory: '/home/yicong/桌面/nnUNet_Folders/nnUNet_preprocessed/Task501_ProstateSegmentation' won't change
        :param batch_dice: True  questons regard this
        :param stage: 1  should be deprecated
        :param unpack_data: True just remain
        :param deterministic: False can be True
        :param fp16: True can be False
        '''
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500
        self.patience = 100
        self.initial_lr = 1e-5
        self.model_name = None
        self.splits_path = splits_path
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        # for auxiliary_task
        self.sampling_method = sampling_method
        
        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        # self.was_initialized = False from parent class
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder) # os.makedirs(directory, exist_ok=True)
            if force_load_plans or (self.plans is None):
                self.load_plans_file() # self.plans = load_pickle(self.plans_file)

            self.process_plans(self.plans)  # Just let it process, we will only use a part of parameters

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            # net_numpool = nnFormer's number of ouputs of the network

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            if not self.self_supervised:
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            # We do not need to specify loss function when using self_supervised
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=True)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=True)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True


    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore
        :return:
        """
        # need to modify
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def initialize_network(self):
        """
        The idea is to use our own model instead of generic 3d
        """
        """
        The parameter settings are in nnformer not from hyperparameters
        """
        if self.model_name == '3d_nnFormer':
            self.network = nnformer(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnConv':
            self.network = nnConv(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_75m':
            self.network = nnformer_75m(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_300m':
            self.network = nnformer_300m(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_pool':
            self.network = nnformer_pool(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_LNOff':
            self.network = nnformer_LNOff(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_noPos':
            self.network = nnformer_noPos(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_absolute':
            self.network = nnformer_absolute(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_sinusoid':
            self.network = nnformer_sinusoid(self.num_input_channels, self.num_classes, deep_supervision=True)
        elif self.model_name == '3d_nnFormer_MAE':
            self.network = nnformer_mae(self.num_input_channels, 1, deep_supervision=False)
        elif self.model_name == '3d_nnConv_MAE':
            self.network = nnConv_mae(self.num_input_channels, 1, deep_supervision=False)
        elif self.model_name == '3d_nnFormer_noPos_MAE':
            self.network = nnformer_noPos_mae(self.num_input_channels, 1, deep_supervision=False)
        elif self.model_name == '3d_nnFormer_sinusoid_MAE':
            self.network = nnformer_sinusoid_mae(self.num_input_channels, 1, deep_supervision=False)
        elif self.model_name == '3d_nnFormer_pool_MAE':
            self.network = nnformer_pool_mae(self.num_input_channels, 1, deep_supervision=False)
        elif self.model_name == '3d_nnFormer_auxiliary':
            if not self.sampling_method:
                self.network = nnformer_auxiliary(self.num_input_channels, self.num_classes, deep_supervision=True)
            else:
                self.network = nnformer_auxiliary(self.num_input_channels, self.num_classes, deep_supervision=True, sampling_method=self.sampling_method)
        elif self.model_name == '3d_nnFormer1_auxiliary':
            self.network = nnformer_auxiliary(self.num_input_channels, self.num_classes, deep_supervision=True,
                                              sampling_method=self.sampling_method)
                

        
            
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=self.weight_decay, amsgrad=False)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability
        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']   
        data = maybe_to_torch(data)
        auxiliary = (self.model_name[self.model_name.rfind('_')+1:]=='auxiliary')
        if not self.self_supervised:
            target = data_dict['target']
            target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            if not self.self_supervised:
                target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                if not auxiliary:
                    output = self.network(data)
                else:
                    output, drloc_outputs = self.network(data)
                del data
                if not self.self_supervised:
                    l = self.loss(output, target)
                else:
                    # The output of our nnformer_mae is the loss
                    l = output 
                if auxiliary:
                    loss_ssup, ssup_items = cal_selfsupervised_loss(drloc_outputs, 0.5)
                    l += loss_ssup
                    
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            if not self.self_supervised:
                l = self.loss(output, target)
            else:
                # The output of our nnformer_mae is the loss
                l = output
            if auxiliary:
                loss_ssup, ssup_items = cal_selfsupervised_loss(drloc_outputs, 0.5)
                l += loss_ssup

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        if not self.self_supervised:
            del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, self.splits_path)

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1
        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)
        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))


    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr
        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret






#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import nnunet
from nnunet.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.summarize_plans import summarize_plans
from nnunet.training.model_restore import recursive_find_python_class


def get_configuration_from_output_folder(folder):
    # split off network_training_output_dir
    folder = folder[len(network_training_output_dir):]
    if folder.startswith("/"):
        folder = folder[1:]

    configuration, task, trainer_and_plans_identifier = folder.split("/")
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    return configuration, task, trainer, plans_identifier


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=(nnunet.__path__[0], "training", "network_training"),
                              base_module='nnunet.training.network_training'):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres', '3d_nnFormer','3d_nnFormer_30m','3d_nnFormer_30m_MRI','3d_nnFormer_75m','3d_nnConv','3d_nnConv_30m', '3d_nnConv_30m_MRI','3d_nnConv_75m', '3d_nnConv_75m_1', '3d_nnFormer_75m_1','3d_nnFormer_75m','3d_nnFormer_300m','3d_nnFormer_pool','3d_nnFormer_AvgPool','3d_nnFormer_AvgPool1','3d_nnFormer_LNOff'
                       ,'3d_nnFormer_absolute','3d_nnFormer_noPos','3d_nnFormer_MAE','3d_nnConv_MAE','3d_nnFormer_absolute_MAE','3d_nnFormer_noPos_MAE','3d_nnFormer_sinusoid','3d_nnFormer_sinusoid_1','3d_nnFormer_sinusoid_2',
                       '3d_nnFormer_sinusoid_MAE','3d_nnFormer_pool_MAE','3d_nnFormer_auxiliary','3d_nnFormer1_auxiliary','3d_nnFormer_MaskImage_MAE']


    dataset_directory = join(preprocessing_output_dir, task)

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")

    plans = load_pickle(plans_file)
    # We use defualt plans(the same as nnUNet3D so this part does not need to be changed
    if (task=='Task501_ProstateSegmentation' or  task=='Task502_ProstateSegmentation' or task=='Task503_ProstateSegmentation' or task=='Task504_LUMCPretrain') and network!='3d_fullres':
        plans['plans_per_stage'][1]['batch_size'] = 4
        plans['plans_per_stage'][1]['patch_size'] = np.array([64, 128, 128])
        plans['plans_per_stage'][1]['pool_op_kernel_sizes']=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
        plans['plans_per_stage'][1]['conv_kernel_sizes']=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
        pickle_file = open(plans_file,'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()
    elif (task=='Task505_PROMISE12' or task == 'Task506_ProstateX') and network!='3d_fullres':
        plans['plans_per_stage'][0]['batch_size'] = 4
        plans['plans_per_stage'][0]['patch_size'] = np.array([16, 128, 128])
        plans['plans_per_stage'][0]['pool_op_kernel_sizes']=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]]
        plans['plans_per_stage'][0]['conv_kernel_sizes']=[[1,3,3],[1,3,3],[3,3,3],[3,3,3],[3,3,3]]
        pickle_file = open(plans_file,'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()
    else:
        plans['plans_per_stage'][1]['batch_size'] = 4
        plans['plans_per_stage'][1]['patch_size'] = np.array([64, 128, 128])
        plans['plans_per_stage'][1]['pool_op_kernel_sizes']=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        plans['plans_per_stage'][1]['conv_kernel_sizes']=[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        pickle_file = open(plans_file,'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()
    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    trainer_class = recursive_find_python_class([join(*search_in)], network_trainer,
                                                current_module=base_module)

    temp_path = network_training_output_dir
    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    print("###############################################")
    print("I am running the following nnUNet: %s" % network)
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class

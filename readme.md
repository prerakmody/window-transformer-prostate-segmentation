**Project_Description**
The Project is built based on nnUNet: The first step is to follow the instructions of the [nnUnet](https://github.com/MIC-DKFZ/nnUNet) to do the Data_Conversion, Experiment Planning and Preprocessing. For this step, you need to store the data in given format in specific folders.

## Installation
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) with python3.8
2. Install [git](https://git-scm.com/downloads)
3. Open a terminal and follow the commands
    - Clone this repository
        - `git clone git@github.com:prerakmody/window-transformer-prostate-segmentation`
    - Create conda env
        ```
        cd window-transformer-prostate-segmentation
        conda deactivate
        conda create --name window-transformer-prostate-segmentation python=3.8
        conda activate window-transformer-prostate-segmentation
        conda develop .  # check for conda.pth file in $ANACONDA_HOME/envs/window-transformer-prostate-segmentation/lib/python3.8/site-packages
        ```
    - Install packages
        - Pytorch
            conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
            ```
        - Other packages
            ```
            pip install scipy seaborn tqdm SimpleITK itk einops timm
            ```
## Train and Inference           
In order to train our own model, we wrote a Trainer that inherits the nnUnetTrainer and saved the models in models folder. Different to the original nnUnet_train command, we have to specify the data_splits filename so as to use our splits.(We rewrote the run_training.py as our main.py to run the training or validation). For example: we use the following command to train the model 3d_nnFormer_LNOff(without Layer Normalization and only train from the second 9 patients from Aahuus Dataset); keep the folder index 0 so that the Trainer use the first folder from splits_file as we modified. Add self-supervised to denote that the task is impainting task which was not supported by the original nnUnet. Also, add load_self_supervised to denote that the weights are loaded form a self-supervised model.

### Data Conversion
Please follow the instruction on [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md. You ought to have three folders:nnUNet_raw_data_base, nnUNet_preprocessed,   RESULTS_FOLDER in which the raw data, preproccessed data and Models are saved. 
1. Put the data in nnUNet_raw_data_base folder
2. Run the following command to generate plans
    - `nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity`


### Datasets
1. 502 denotes the HMC dataset(A).
2. 503 denotes the HMC+LUMC dataset(A+B).
3. 504 denotes the LUMC dataset (B)
4. splits_full, splits_h1, splits_h2 denote using the full, first half and second half of HMC dataset as training set and the other half as validation. We always use the folder 0 because we mannually create the splits(indicated by the original nnunet).

### Experiment1
The experiment1 contains two model: 3d_nnFormer and 3d_nnConv. Model with prefix MAE denotes using self-supervised task to pretrain the model.
```bash
main 3d_nnFormer TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnConv TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnFormer TransformerUNetTrainer 503 splits_h2.pkl 0
main 3d_nnConv TransformerUNetTrainer 503 splits_h2.pkl 0
main 3d_nnFormer TransformerUNetTrainer 504 splits_full.pkl 0
main 3d_nnFormer_pool_MAE TransformerUNetTrainer 504 splits_Pretrain.pkl 0 --self_supervised
main 3d_nnConv TransformerUNetTrainer 504 splits_full.pkl 0
```

Also load the pretrained weights from pretrained models:Supervised and Self-Supervised

```bash
 main 3d_nnFormer TransformerUNetTrainer 502 splits_full.pkl 0 -pretrained_weights Former_Pretrain_LUMC.model
 main 3d_nnConv TransformerUNetTrainer 502 splits_full.pkl 0 -pretrained_weights Conv_Pretrain_LUMC.model
 main 3d_nnConv TransformerUNetTrainer 502 splits_full.pkl 0 --load_self_supervised -pretrained_weights Conv_Pretrain_MAE.model 
 main 3d_nnFormer TransformerUNetTrainer 502 splits_full.pkl 0 --load_self_supervised -pretrained_weights Former_Pretrain_MAE.model 
```

### Experiment2
The experiment2 contains two model: 3d_nnFormer and 3d_nnFormer_pool. The format is the same as the experiment1 with the model name as the only difference.

```bash
main 3d_nnFormer_pool TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnFormer_pool TransformerUNetTrainer 503 splits_h2.pkl 0
main 3d_nnFormer_pool TransformerUNetTrainer 504 splits_pretrain.pkl 0
main 3d_nnFormer_pool_MAE TransformerUNetTrainer 504 splits_pretrain.pkl 0
```

```bash
main 3d_nnFormer_pool TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnFormer_pool TransformerUNetTrainer 502 splits_h1.pkl 0 -pretrained_weights Former_pool_Pretrain_LUMC.model
main 3d_nnFormer_pool TransformerUNetTrainer 502 splits_full.pkl 0 --load_self_supervised -pretrained_weights Former_pool_Pretrain_MAE.model 

```

### Experiment3
The experiment1 contains three model: 3d_nnFormer, 3d_nnFormer_absolute, and 3d_nnFormer_noPos. Note that 3d_nnFormer uses the relative positional bias.

```bash
main 3d_nnFormer_absolute TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnFormer_absolute TransformerUNetTrainer 503 splits_h2.pkl 0
main 3d_nnFormer_absolute TransformerUNetTrainer 504 splits_pretrain.pkl 0
main 3d_nnFormer_absolute_MAE TransformerUNetTrainer 504 splits_pretrain.pkl 0
main 3d_nnFormer_noPos TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnFormer_noPos TransformerUNetTrainer 503 splits_h2.pkl 0
main 3d_nnFormer_noPos TransformerUNetTrainer 504 splits_pretrain.pkl 0
main 3d_nnFormer_noPos_MAE TransformerUNetTrainer 504 splits_pretrain.pkl 0
```

```bash
main 3d_nnFormer_absolute TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnFormer_absolute TransformerUNetTrainer 502 splits_h1.pkl 0 -pretrained_weights Former_pool_Pretrain_LUMC.model
main 3d_nnFormer_absolute TransformerUNetTrainer 502 splits_full.pkl 0 --load_self_supervised -pretrained_weights Former_pool_Pretrain_MAE.model 
main 3d_nnFormer_noPos TransformerUNetTrainer 502 splits_h2.pkl 0
main 3d_nnFormer_noPos TransformerUNetTrainer 502 splits_h1.pkl 0 -pretrained_weights Former_pool_Pretrain_LUMC.model
main 3d_nnFormer_noPos TransformerUNetTrainer 502 splits_full.pkl 0 --load_self_supervised -pretrained_weights Former_pool_Pretrain_MAE.model 

```
# Inference
In order to Inference from our model, we modified the predict_simple.py and made it work for our Trainer. For example, we use the below command to conduct the inference from the 3d_nnFormer_LNOff model. Using nnUNet_predict if you follow the nnUnet instruction to set the environment paths or you can always use predict_simple from nnunet/inference. The Format is the same for all three experiements.
```bash
nnUNet_predict -i DataPath  -o TargetPath -t Task -tr TransformerUNetTrainer -f 0 -m Model
```
```bash
nnUNet_predict -i DataPath  -o TargetPath -t 502 -tr TransformerUNetTrainer -f 0 -m 3d_nnFormer
```





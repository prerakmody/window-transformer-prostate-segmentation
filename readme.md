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
```bash
main 3d_nnFormer_LNOff TransformerUNetTrainer 502 splits_h2.pkl 0
```

```bash
main 3d_nnFormer_MAE TransformerUNetTrainer 504 splits_Pretrain.pkl 0 --self_supervised
```

```bash
main 3d_nnConv TransformerUNetTrainer 502 splits_h1.pkl 0 --load_self_supervised -pretrained_weights Conv_Pretrain_MAE.model 
```

In order to Inference from our model, we modified the predict_simple.py and made it work for our Trainer. For example, we use the below command to conduct the inference from the 3d_nnFormer_LNOff model.
```bash
nnUNet_predict -i DataPath  -o TargetPath -t 502 -tr TransformerUNetTrainer -f 0 -m 3d_nnFormer_LNOff
```





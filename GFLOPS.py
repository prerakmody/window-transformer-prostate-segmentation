import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from nnunet.network_architecture.models.nnConv import nnConv
from nnunet.network_architecture.models.nnFormer import nnformer
with torch.cuda.device(0):
    net1 = nnformer(1, 5, deep_supervision=True)
    net2 = nnConv(1, 5, deep_supervision=True)
    flops, params = get_model_complexity_info(net1, (1, 64, 128, 128), as_strings=True,
                                              print_per_layer_stat=False)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:  ' + flops)
    print('Params: ' + params)

    flops, params = get_model_complexity_info(net2, (1, 64, 128, 128), as_strings=True,
                                              print_per_layer_stat=False)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:  ' + flops)
    print('Params: ' + params)


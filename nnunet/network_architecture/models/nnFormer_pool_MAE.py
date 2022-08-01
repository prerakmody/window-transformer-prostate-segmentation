from einops import rearrange
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

#Replace self-attention with CNN kernel_size=3

class SinglePool3DBlock(nn.Module):
    #use for resolution recover
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, Pool3D=nn.MaxPool3d, norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.conv = Pool3D(kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2))
        self.act = act_layer()
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,4,1)
        x = self.norm(x)
        x = x.permute(0,4,1,2,3)
        x = self.act(x)
        return x

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OperationBlock(nn.Module):
    '''
    Assume that Convolution can replace Transformer block
    '''

    def __init__(self, dim, input_resolution, kernel_size = 5, stride = 1,
                 mlp_ratio=5., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.operation_block = SinglePool3DBlock(dim, dim, kernel_size=kernel_size, stride=stride)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, C, L = x.shape
        S, H, W = self.input_resolution

        assert L == S * H * W, "input feature has wrong size"

        x = x.permute(0,2,1)
        shortcut = x
        x = self.norm1(x)
        x = x.permute(0,2,1)
        x = x.view(B, C, S, H, W)

        x = self.operation_block(x)

        x = x.permute(0, 2, 3, 4, 1).view(B, S*H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.permute(0,2,1)


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=2, stride=2)
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, 2 * C)
        return x


class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up = nn.ConvTranspose3d(dim, dim // 2, 2, 2)

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.up(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, C // 2)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 use_checkpoint=False):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            OperationBlock(
                dim=dim,
                input_resolution = input_resolution,
                kernel_size=3,
                stride=1,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)
        if self.downsample is not None:
            x = x.permute(0,2,1)
            x_down = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                 ):
        super().__init__()
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            OperationBlock(
                dim=dim,
                input_resolution=input_resolution,
                kernel_size=3,
                stride=1,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer

        self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer)

    def forward(self, x, skip, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        x_up = self.Upsample(x, S, H, W)

        x_up += skip
        S, H, W = S * 2, H * 2, W * 2
        x_up = x_up.permute(0, 2, 1)
        for blk in self.blocks:
            x_up = blk(x_up)

        return x_up, S, H, W


# done
class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj1 = project(in_chans, embed_dim // 2, [1, 2, 2], 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim // 2, embed_dim, [2, 2, 2], 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))

        x = self.proj1(x)

        x = self.proj2(x)

        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Ws, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        #mask_toekn
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)
        x_masked = torch.cat([x_masked[:, :, :], mask_tokens], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        return x_masked, mask, ids_restore

    def forward(self, x):
        """Forward function."""

        x = self.patch_embed(x)
        down = []

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Ws, Wh, Ww), align_corners=True,
                                               mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)

        # mask the 30% percent of patches
        x, mask, ids_restore = self.random_masking(x, 0.3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = x.permute(0, 2, 1)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()

                down.append(out)
        return down, mask, ids_restore

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2, 2, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[2] // patch_size[2] // 2 ** (len(depths) - i_layer - 1)),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):

        outs = []
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose(1, 2)
            skips[index] = i
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]

            x, S, H, W, = layer(x, skips[i], S, H, W)
            out = x.view(-1, self.num_features[i], S, H, W)
            x = x.permute(0,2,1)
            outs.append(out)
        return outs


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)

    def forward(self, x):
        x = self.up(x)

        return x


class nnformer_pool_mae(SegmentationNetwork):

    def __init__(self, input_channels, num_classes, deep_supervision=True):

        super(nnformer_pool_mae, self).__init__()

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = nn.Conv3d

        self.upscale_logits_ops = []

        self.upscale_logits_ops.append(lambda x: x)

        embed_dim = 192
        pretrain_img_size = [64, 128, 128]
        depths = [2, 2, 2, 2]
        num_heads = [6, 12, 24, 48]
        patch_size = [2, 4, 4]
        window_size = 4
        self.model_down = SwinTransformer(pretrain_img_size=pretrain_img_size,
                                          embed_dim=embed_dim, patch_size=patch_size, depths=depths,
                                          in_chans=input_channels)
        # for some reasons,the decoder is named with encoder
        self.encoder = encoder(pretrain_img_size=pretrain_img_size, embed_dim=embed_dim,
                               patch_size=patch_size, depths=depths[::-1][1:])

        self.final = []
        for i in range(len(depths) - 1):
            self.final.append(final_patch_expanding(embed_dim * 2 ** i, num_classes, patch_size=patch_size))
        self.final = nn.ModuleList(self.final)

    def CT_patchify(self, imgs):
        """
        imgs: (N, 1, D, H, W)
        x: (N, L, patch_size**3 * 1)
        """
        # This is the fixed setting for my experiment
        p = [2, 4, 4]
        d, h, w = imgs.shape[2] // p[0], imgs.shape[3] // p[1], imgs.shape[4] // p[2]
        x = imgs.reshape(shape=(imgs.shape[0], 1, d, p[0], h, p[1], w, p[2]))
        x = torch.einsum('ncdkhpwq->ndhwkpqc', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p[0] * p[1] * p[2]))
        return x

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=True):
        """
        imgs: [N, 1, D, H, W]
        pred: [N, 1, D, H, W]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.CT_patchify(imgs)
        pred = self.CT_patchify(pred)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    def forward(self, x):

        seg_outputs = []
        skips, mask, ids_restore = self.model_down(x)
        neck = skips[-1]

        out = self.encoder(neck, skips)

        for i in range(len(out)):
            seg_outputs.append(self.final[-(i + 1)](out[i]))

        pred = seg_outputs[-1]
        loss = self.forward_loss(x, pred, mask)
        return loss



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

class local_global_fuse(nn.Module):
    '''
    fuse the global and local information
    '''
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.local2global = nn.Linear(window_size**3, 1, bias=True)
    def forward(self,x, S, H, W, global_x):
        '''

        :param x: local feature (B, S*H*W, C)
        :param global_x: (B, GS*GH*GW, C)
        :return:
        '''
        B, L, C = x.shape
        assert L == S * H * W, "input feature has wrong size"

        # Fuse global and local information

        x = x.view(B, S, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(B, -1, self.window_size * self.window_size * self.window_size, C) #for example (4,512,64,192)

        global_message = global_x[:,:,None,:]
        x += global_message

        local_message = self.local2global(x.permute(0,1,3,2))
        global_x += torch.squeeze(local_message,dim=3)

        return x, global_x



def window_partition(x, window_size, is_global=False):
    if is_global:
        B, S, H, W, C  = x.shape
        x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, S // window_size, H // window_size, W // window_size, -1)
        return x

    else:
        B, S, H, W, C = x.shape
        x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
        return windows

def window_reverse(windows, window_size, S, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x


class GLAttention(nn.Module):
    """ Local Window based multi-head self attention (W-MSA) module with relative position bias.
        Global based multi-head self attention (G-MSA) module with relative position bias.
        The parameters are shared between them in Q,K,V linear transformation while the position bias is different because of different size

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, global_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.global_size = global_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)*(2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)


        # define a parameter table of relative position bias for Global_MSA
        self.global_relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.global_size[0] - 1) * (2 * self.global_size[1] - 1) * (2 * self.global_size[2] - 1),num_heads))

        # get pair-wise relative position index for each token in global feature map
        coords_s = torch.arange(self.global_size[0])
        coords_h = torch.arange(self.global_size[1])
        coords_w = torch.arange(self.global_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.global_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.global_size[1] - 1
        relative_coords[:, :, 2] += self.global_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.global_size[1] - 1)*(2 * self.global_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.global_size[2] - 1

        global_relative_position_index = relative_coords.sum(-1)
        self.register_buffer("global_relative_position_index", global_relative_position_index)


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, is_global):
        """ Forward function.

        Args:
            x: input features either global or window-based
            is_global: decides Global Attention or Local Attention

        """
        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        if is_global:
            relative_position_bias = self.global_relative_position_bias_table[self.global_relative_position_index.view(-1)].view(
                self.global_size[0] * self.global_size[1] * self.global_size[2],
                self.global_size[0] * self.global_size[1] * self.global_size[2], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        else:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GLTransformerBlock(nn.Module):
    """ Window Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, global_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.global_resolution = global_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = GLAttention(
            dim, window_size=to_3tuple(self.window_size), global_size=global_resolution, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.fuse = local_global_fuse(window_size=self.window_size)

    def forward(self, x, global_x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"

        B ,GL, C = global_x.shape
        GS, GH, GW = self.global_resolution
        assert GL == GS * GH * GW, "global input feature has wrong size"

        # Window-Attention for local-tokens
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)


        # pad feature maps to multiples of window size
        # Luckily we do not have to do it for global one
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape


        # partition windows
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,C)
        # W-MSA
        attn_windows = self.attn(x_windows, is_global=False)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)
        x = shifted_x
        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # compute global attention is much easier
        global_shortcut = global_x
        global_x = self.norm1(global_x)
        global_x = self.attn(global_x, is_global=True)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        global_x = global_shortcut + self.drop_path(global_x)
        global_x = global_x + self.drop_path((self.mlp(self.norm2(global_x))))

        # Fuse global and local information
        x, global_x = self.fuse(x, S, H, W, global_x)
        x = x.view(B, S * H * W, C)


        return x, global_x


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
                 global_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            GLTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                global_resolution=global_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W, global_x, GS, GH, GW, last=False):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """


        for blk in self.blocks:
            blk.H, blk.W = H, W
            x, global_x = blk(x, global_x) #

        #oh my ugly code!!!!!
        if not last and self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            global_x_down = self.downsample(global_x, GS, GH, GW)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            GWs, GWh, GWw = (GS + 1) // 2, (GH + 1) // 2, (GW + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww, global_x, GS, GH, GW, global_x_down, GWs, GWh, GWw
        else:
            return x, S, H, W, x, S, H, W, global_x, GS, GH, GW, global_x, GS, GH, GW


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
                 global_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            GLTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                global_resolution=global_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer

        self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer)

    def forward(self, x, skip, S, H, W, global_x, global_skip, GS, GH, GW):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        x_up = self.Upsample(x, S, H, W)
        global_x_up = self.Upsample(global_x, GS, GH, GW)

        x_up += skip
        global_x_up += global_skip

        S, H, W = S * 2, H * 2, W * 2
        GS, GH, GW = GS * 2, GH * 2, GW * 2

        for blk in self.blocks:
            x_up, global_x_up = blk(x_up, global_x_up)

        return x_up, S, H, W, global_x_up, GS, GH, GW


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

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 global_patch_size=4,
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 global_window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.global_window_size = global_window_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.global_conv = nn.Conv3d(embed_dim, embed_dim, global_patch_size, global_patch_size, groups=embed_dim)

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
                global_resolution=(
                    pretrain_img_size[0]//global_patch_size // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1]//global_patch_size // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2]//global_patch_size // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
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

    def forward(self, x):
        """Forward function."""


        x = self.patch_embed(x)
        global_x = self.global_conv(x)
        down = []
        global_down = []

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        GWs, GWh, GWw = global_x.size(2), global_x.size(3), global_x.size(4)

        x = self.pos_drop(x)
        global_x = self.pos_drop(global_x)

        x = x.flatten(2).transpose(1, 2)
        global_x = global_x.flatten(2).transpose(1, 2)

        for i in range(self.num_layers):
            layer = self.layers[i]
            if i==self.num_layers-1:
                x_out, S, H, W, x, Ws, Wh, Ww, global_x_out, GS, GH, GW, global_x, GWs, GWh, GWw = layer(x, Ws, Wh, Ww, global_x, GWs, GWh, GWw, last=True)
            else:
                x_out, S, H, W, x, Ws, Wh, Ww, global_x_out, GS, GH, GW, global_x, GWs, GWh, GWw = layer(x, Ws, Wh, Ww, global_x, GWs, GWh, GWw, last=False)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                global_x_out = norm_layer(global_x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                global_out = global_x_out.view(-1, GS, GH, GW, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                down.append(out)
                global_down.append(global_out)
        return down, global_down

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 global_patch_size=4,
                 depths=[2, 2, 2],
                 num_heads=[24, 12, 6],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
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
                global_resolution=(
                    pretrain_img_size[0] // global_patch_size // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // global_patch_size // patch_size[1] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[2] // global_patch_size // patch_size[2] // 2 ** (len(depths) - i_layer - 1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, global_x, skips, global_skips):

        outs = []
        global_outs = []
        S, H, W = x.size(2), x.size(3), x.size(4)
        GS, GH, GW = global_x.size(2), global_x.size(3), global_x.size(4)

        x = x.flatten(2).transpose(1, 2)
        global_x = global_x.flatten(2).transpose(1,2)
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose(1, 2)
            skips[index] = i

        for index, i in enumerate(global_skips):
            i = i.flatten(2).transpose(1, 2)
            global_skips[index] = i
        x = self.pos_drop(x)
        global_x = self.pos_drop(global_x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, S, H, W, global_x, GS, GH, GW = layer(x, skips[i], S, H, W, global_x, global_skips[i], GS, GH, GW)
            out = x.view(-1, S, H, W, self.num_features[i])
            global_out = global_x.view(-1, GS, GH, GW, self.num_features[i])
            outs.append(out)
            global_outs.append(global_out)
        return outs, global_outs


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.up(x)

        return x


class uuformer_v2(SegmentationNetwork):

    def __init__(self, input_channels, num_classes, deep_supervision=True):

        super(uuformer_v2, self).__init__()

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
        self.model_down = SwinTransformer(pretrain_img_size=pretrain_img_size, window_size=window_size,
                                          embed_dim=embed_dim, patch_size=patch_size, depths=depths,
                                          num_heads=num_heads, in_chans=input_channels)
        # for some reasons,the decoder is named with encoder
        self.encoder = encoder(pretrain_img_size=pretrain_img_size, embed_dim=embed_dim, window_size=window_size,
                               patch_size=patch_size, num_heads=num_heads[::-1][1:], depths=depths[::-1][1:])

        self.final = []
        for i in range(len(depths) - 1):
            self.final.append(final_patch_expanding(embed_dim * 2 ** i, num_classes, patch_size=patch_size))
        self.final = nn.ModuleList(self.final)

    def forward(self, x):

        seg_outputs = []
        skips, global_skips = self.model_down(x)
        neck, global_neck = skips[-1], global_skips[-1]

        out, global_out = self.encoder(neck, global_neck, skips, global_skips)

        for i in range(len(out)):
            seg_outputs.append(self.final[-(i + 1)](out[i]))

        if self._deep_supervision and self.do_ds:
            deep_supervision_result = [seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
            deep_supervision_result.append(self.final[0](global_out[-1]))
            return tuple(deep_supervision_result)
        else:

            return seg_outputs[-1]


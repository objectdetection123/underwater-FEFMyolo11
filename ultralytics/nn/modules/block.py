# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
from typing import Tuple, Optional, List
from torch import nn, Tensor, LongTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
try:
    from mmcv.cnn import build_activation_layer, build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
    from mmengine.model import constant_init, normal_init
except ImportError as e:
    pass

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown","SPPF_LSKA",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

#####################################改进


# class SP(nn.Module) :
#     def __init__(self, k=3, s=1) :
#         super(SP, self).__init__()
#         self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

#     def forward(self, x) :
#         return self.m(x)


# class SPPF(nn.Module) :
#     # spp-elan
#     def __init__(self, c1, c2, c3) :  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = SP(5)
#         self.cv3 = SP(5)
#         self.cv4 = SP(5)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)

#     def forward(self, x) :
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         return self.cv5(torch.cat(y, 1))
####################
class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

########################################

class CAA(nn.Module) :
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None :
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x) :
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x


class GhostConv(nn.Module) :
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True) :
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x) :
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)




class SimAM(torch.nn.Module) :
    def __init__(self, channels=None, e_lambda=1e-4) :
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self) :
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name() :
        return "simam"

    def forward(self, x) :
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class lightC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            lightC3k(self.c, self.c, 2, shortcut, g) if c3k else improveBottleneckWithdwconvWithodconvwitham22(self.c, self.c, shortcut, g) for _ in range(n)
        )


# ============= 新增的轻量高效模块 =============
class GhostFeatureFusion(nn.Module) :
    """Ghost特征融合模块 (零参数)"""

    def __init__(self, channels) :
        super().__init__()
        self.channels = channels

    def forward(self, x) :
        # 使用特征自身生成注意力图
        spatial_avg = x.mean(dim=1, keepdim=True)
        spatial_max, _ = x.max(dim=1, keepdim=True)
        spatial_att = spatial_avg + spatial_max

        # 使用Ghost操作生成特征
        ghost_feat = x * torch.sigmoid(spatial_att)

        # 特征融合
        return 0.5 * x + 0.5 * ghost_feat


class MetaAttention(nn.Module) :
    """元注意力机制 (极轻量)"""

    def __init__(self, channels, reduction=16) :
        super().__init__()
        self.reduced_dim = max(4, channels // reduction)

        # 极简通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_dim, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力 (无参数)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x) :
        # 通道注意力
        channel_att = self.channel_att(x)

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))

        # 双重注意力融合
        return x * channel_att * spatial_att


class ChannelRescale(nn.Module) :
    """动态通道重标定 (轻量但高效)"""

    def __init__(self, channels) :
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, 4),  # 固定小尺寸
            nn.ReLU(inplace=True),
            nn.Linear(4, channels),
            nn.Sigmoid()
        )

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x) :
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # 动态重标定
        return x * y * self.scale




class UltraLightRefinementModule(nn.Module) :
    """极轻量特征精炼模块 (ULRM)
    1. 零参数空间注意力机制
    2. 轻量通道重校准
    3. 自适应特征融合
    """

    def __init__(self, channels, reduction=16) :
        super().__init__()
        # 零参数空间注意力
        self.spatial_att = nn.Identity()  # 无参数操作

        # 轻量通道重校准 (仅增加channels/reduction参数)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 自适应融合权重 (单个可学习参数)
        self.fusion_weight = nn.Parameter(torch.tensor([0.7]), requires_grad=True)

    def forward(self, x) :
        # 零参数空间注意力
        spatial_att = self.calculate_spatial_attention(x)

        # 通道重校准
        channel_att = self.channel_att(x)

        # 自适应融合
        return self.fusion_weight * (x * spatial_att) + (1 - self.fusion_weight) * (x * channel_att)

    def calculate_spatial_attention(self, x) :
        """零参数空间注意力机制"""
        # 计算空间重要性
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        avg_val = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(max_val + avg_val)
        return spatial_att

class DWConv(nn.Module) :
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu") :
        super().__init__()
        self.dconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=ksize,
            stride=stride, padding=ksize // 2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act == "silu" else nn.ReLU()

    def forward(self, x) :
        x = self.act(self.bn1(self.dconv(x)))
        return self.act(self.bn2(self.pconv(x)))


class ODConv2d_3rd(nn.Conv2d) :
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 K=4, r=1 / 16, save_parameters=False,
                 padding_mode='zeros', device=None, dtype=None) :
        factory_kwargs = {'device' : device, 'dtype' : dtype}
        self.K = K
        self.r = r
        self.save_parameters = save_parameters

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        del self.weight
        self.weight = nn.Parameter(torch.empty((
            K,
            out_channels,
            in_channels // groups,
            *self.kernel_size,
        ), **factory_kwargs))

        if bias :
            del self.bias
            self.bias = nn.Parameter(torch.empty(K, out_channels, **factory_kwargs))

        hidden_dim = max(int(in_channels * r), 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)

        self.fc_f = nn.Linear(hidden_dim, out_channels)
        if not save_parameters or self.kernel_size[0] * self.kernel_size[1] > 1 :
            self.fc_s = nn.Linear(hidden_dim, self.kernel_size[0] * self.kernel_size[1])
        if not save_parameters or in_channels // groups > 1 :
            self.fc_c = nn.Linear(hidden_dim, in_channels // groups)
        if not save_parameters or K > 1 :
            self.fc_w = nn.Linear(hidden_dim, K)

        self.reset_parameters()

    def reset_parameters(self) :
        fan_out = self.kernel_size[0] * self.kernel_size[1] * self.out_channels // self.groups
        for i in range(self.K) :
            self.weight.data[i].normal_(0, math.sqrt(2.0 / fan_out))
        if self.bias is not None :
            self.bias.data.zero_()

    def extra_repr(self) :
        return super().extra_repr() + f', K={self.K}, r={self.r:.4}'

    def get_weight_bias(self, context) :
        B, C, H, W = context.shape

        if C != self.in_channels :
            raise ValueError(
                f"Expected context{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        x = self.gap(context)
        x = self.fc(x)
        if x.size(0) > 1 :
            x = self.bn(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.act(x)

        attn_f = self.fc_f(x).sigmoid()
        attn = attn_f.view(B, 1, -1, 1, 1, 1)
        if hasattr(self, 'fc_s') :
            attn_s = self.fc_s(x).sigmoid()
            attn = attn * attn_s.view(B, 1, 1, 1, *self.kernel_size)
        if hasattr(self, 'fc_c') :
            attn_c = self.fc_c(x).sigmoid()
            attn = attn * attn_c.view(B, 1, 1, -1, 1, 1)
        if hasattr(self, 'fc_w') :
            attn_w = self.fc_w(x).softmax(-1)
            attn = attn * attn_w.view(B, -1, 1, 1, 1, 1)

        weight = (attn * self.weight).sum(1)
        weight = weight.view(-1, self.in_channels // self.groups, *self.kernel_size)

        bias = None
        if self.bias is not None :
            if hasattr(self, 'fc_w') :
                bias = attn_w @ self.bias
            else :
                bias = self.bias.tile(B, 1)
            bias = bias.view(-1)

        return weight, bias

    def forward(self, input, context=None) :
        B, C, H, W = input.shape

        if C != self.in_channels :
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        weight, bias = self.get_weight_bias(context or input)

        output = nn.functional.conv2d(
            input.view(1, B * C, H, W), weight, bias,
            self.stride, self.padding, self.dilation, B * self.groups)
        output = output.view(B, self.out_channels, *output.shape[2 :])

        return output


class ODConv_3rd(nn.Module) :
    def __init__(self, c1, c2, k=1, s=1, kerNums=1, g=1, p=None, act=True) :
        super().__init__()
        self.conv = ODConv2d_3rd(c1, c2, k, s, autopad(k, p), groups=g, K=kerNums)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x) :
        return self.act(self.bn(self.conv(x)))


import torch.nn as nn
import torch
import math
class ECA(nn.Module):
    """Efficient Channel Attention Module."""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                             padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class improveBottleneckWithdwconvWithodconvwitham22(nn.Module) :
    """改进的bottleneck:
    1. 第一个卷积使用DWConv (深度可分离卷积)
    2. 第二个卷积使用DSConv (深度可分离卷积变体)
    3. 输出拼接后使用残差连接
    4. 在最终输出前应用ECA注意力
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, kerNums=2) :
        super().__init__()
        # 计算中间通道数
        c_ = max(8, int(c2 * e))  # 确保最小通道数

        # 第一个卷积使用DWConv (深度可分离卷积)
        self.dwconv = DWConv(c1, c_, ksize=k[0])

        # 第二个卷积使用DSConv
        self.dsconv = DSConv2D(c_, c_, k=k[1], s=1, g=g)

        # 通道调整层 (用于残差连接)
        self.channel_adjust = nn.Identity()
        if c1 != c2 :
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2)
            )

        # 拼接后通道数
        concat_channels = c_ * 2

        # 输出调整层 (将拼接后的特征调整到目标通道数)
        self.output_adjust = nn.Sequential(
            nn.Conv2d(concat_channels, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        # ECA注意力放在最终输出前 (通道数为c2)
        self.eca = ECA(c2)

        # 残差连接条件
        self.shortcut = shortcut

    def forward(self, x) :
        # 保存原始输入用于残差连接
        identity = self.channel_adjust(x) if self.shortcut else 0

        # 通过DWConv
        x_dw = self.dwconv(x)

        # 通过DSConv
        x_ds = self.dsconv(x_dw)

        # 拼接两个输出 (通道维度)
        x_cat = torch.cat([x_dw, x_ds], dim=1)

        # 调整输出通道数
        x_out = self.output_adjust(x_cat)

        # 添加残差连接
        if self.shortcut :
            x_out = x_out + identity

        # 在最终输出前应用ECA注意力
        return self.eca(x_out)



############################

class PSAWithSPPF(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.c = c
        self.attn = ECA(c, k_size=3)
        
        # 改进的SPPF分支
        self.sppf_conv_reduce = DSConv2D(c, c // 2, 1, 1)  # 减少通道数
        
        # 三个不同尺度的池化层
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(5, 1, 2)
        self.pool3 = nn.MaxPool2d(5, 1, 2)
        
        # 确保输出通道数与输入一致
        self.sppf_conv_expand = DSConv2D(c // 2 * 4, c, 1, 1)
        
        # 轻量化的FFN
        self.ffn = nn.Sequential(
            DSConv2D(c, c * 2, 1),
            DSConv2D(c * 2, c, 1, act=False)
        )
        self.add = shortcut

    def forward(self, x):
        # 注意力分支
        attn_out = self.attn(x)
        
        # SPPF多尺度分支
        reduced = self.sppf_conv_reduce(x)
        
        # 应用三个不同尺度的池化
        p1 = self.pool1(reduced)
        p2 = self.pool2(p1)
        p3 = self.pool3(p2)
        
        # 拼接多尺度特征
        sppf_out = torch.cat([reduced, p1, p2, p3], dim=1)
        sppf_out = self.sppf_conv_expand(sppf_out)
        
        # 融合两个分支
        if self.add:
            x = x + attn_out + sppf_out
        else:
            x = attn_out + sppf_out
        
        # FFN处理
        if self.add:
            x = x + self.ffn(x)
        else:
            x = self.ffn(x)
        return x


class C2PSA3(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        
        # 使用DSConv2D替代标准Conv
        self.cv1 = DSConv2D(c1, 2 * self.c, 1, 1)
        self.cv2 = DSConv2D(2 * self.c, c1, 1)
        
        # 使用改进的PSAWithSPPF模块
        self.m = nn.Sequential(*(PSAWithSPPF(
            self.c, attn_ratio=0.5, num_heads=self.c // 64
        ) for _ in range(n)))

    def forward(self, x):
        # 分割通道
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        # 处理b分支
        b = self.m(b)
        
        # 合并分支
        return self.cv2(torch.cat((a, b), 1))
#######################
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
class TopkRouting(nn.Module) :
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False) :
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor] :
        if not self.diff_routing :
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index


class KVGather(nn.Module) :
    def __init__(self, mul_weight='none') :
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: torch.Tensor, r_weight: torch.Tensor, kv: torch.Tensor) :
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv))
        if self.mul_weight == 'soft' :
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        elif self.mul_weight == 'hard' :
            raise NotImplementedError('differentiable hard routing TBA')
        return topk_kv


class QKVLinear(nn.Module) :
    def __init__(self, dim, qk_dim, bias=True) :
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x) :
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module) :
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3, auto_pad=False) :
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0
        self.scale = qk_scale or self.qk_dim ** -0.5

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x : torch.zeros_like(x)

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing

        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)

        if self.soft_routing :
            mul_weight = 'soft'
        elif self.diff_routing :
            mul_weight = 'hard'
        else :
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        self.param_attention = param_attention
        if self.param_attention == 'qkvo' :
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv' :
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else :
            raise ValueError(f'param_attention mode {self.param_attention} is not supported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel

        if self.kv_downsample_mode == 'ada_avgpool' :
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool' :
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool' :
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool' :
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity' :
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool' :
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv' :
            raise NotImplementedError('conv policy is not implemented yet!')
        else :
            raise ValueError(f'kv_down_sample_mode {self.kv_downsample_mode} is not supported!')

        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad

    def forward(self, x) :
        # 输入格式转换: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)

        if self.auto_pad :
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.size()
        else :
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0

        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
        q, kv = self.qkv(x)

        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0 :self.qk_dim].mean([2, 3])

        lepe = self.lepe(rearrange(kv[..., self.qk_dim :], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        r_weight, r_idx = self.router(q_win, k_win)

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        out = self.wo(out)

        if self.auto_pad and (pad_r > 0 or pad_b > 0) :
            out = out[:, :H_in, :W_in, :].contiguous()

        # 输出格式转换: (N, H, W, C) -> (N, C, H, W)
        return out.permute(0, 3, 1, 2)


class EnhancedC2PSA3(nn.Module) :
    def __init__(self, c1, c2, n=1, e=0.5, bra_params=None) :
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 设置BRA的默认参数
        if bra_params is None :
            bra_params = {
                'dim' : self.c,
                'num_heads' : max(1, self.c // 64),
                'n_win' : 7,
                'topk' : 4,
                'side_dwconv' : 3,
                'auto_pad' : True
            }

        # 使用BRA替换PSABlock
        self.m = nn.Sequential(*(BiLevelRoutingAttention(**bra_params) for _ in range(n)))

    def forward(self, x) :
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
######################



# # import torch
# # import torch.nn as nn
# # import math


# # class DSConv2D(nn.Module) :
# #     """深度可分离卷积实现"""

# #     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, act=True) :
# #         super().__init__()
# #         self.depthwise = nn.Conv2d(
# #             in_channels, in_channels, kernel_size, stride,
# #             padding, dilation, groups=in_channels, bias=False
# #         )
# #         self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)
# #         self.bn = nn.BatchNorm2d(out_channels)
# #         self.act = nn.SiLU() if act else nn.Identity()

# #     def forward(self, x) :
# #         x = self.depthwise(x)
# #         x = self.pointwise(x)
# #         x = self.bn(x)
# #         return self.act(x)


# # class PConv(nn.Module) :
# #     """部分卷积(PConv)实现，基于您提供的代码"""

# #     def __init__(self, dim, ouc, n_div=4, forward='split_cat') :
# #         super().__init__()
# #         self.dim_conv3 = dim // n_div
# #         self.dim_untouched = dim - self.dim_conv3
# #         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

# #         # 使用DSConv2D替代标准卷积
# #         self.conv = DSConv2D(dim, ouc, k=1)

# #         if forward == 'slicing' :
# #             self.forward = self.forward_slicing
# #         elif forward == 'split_cat' :
# #             self.forward = self.forward_split_cat
# #         else :
# #             raise NotImplementedError

# #     def forward_slicing(self, x) :
# #         # only for inference
# #         x = x.clone()
# #         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
# #         x = self.conv(x)
# #         return x

# #     def forward_split_cat(self, x) :
# #         # for training/inference
# #         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
# #         x1 = self.partial_conv3(x1)
# #         x = torch.cat((x1, x2), 1)
# #         x = self.conv(x)
# #         return x


# class EfficientChannelAttention(nn.Module) :
#     """高效通道注意力机制"""

#     def __init__(self, channels, gamma=2, b=1) :
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # 动态计算卷积核大小
#         t = int(abs((math.log(channels, 2) + b) / gamma))
#         k = t if t % 2 else t + 1

#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x) :
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, 1, c)
#         y = self.conv(y)
#         y = self.sigmoid(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class AdaptiveMultiScalePooling(nn.Module) :
#     """自适应多尺度池化模块，结合PConv思想"""

#     def __init__(self, channels, n_div=4) :
#         super().__init__()
#         self.channels = channels
#         self.n_div = n_div

#         # 使用PConv替代标准卷积
#         self.pconv = PConv(channels, channels, n_div=n_div)

#         # 自适应选择池化核大小
#         self.adaptive_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, max(channels // 4, 8), 1),
#             nn.ReLU(),
#             nn.Conv2d(max(channels // 4, 8), 3, 1),
#             nn.Softmax(dim=1)
#         )

#         # 不同尺度的池化层
#         self.pool3 = nn.MaxPool2d(3, 1, 1)
#         self.pool5 = nn.MaxPool2d(5, 1, 2)
#         self.pool7 = nn.MaxPool2d(7, 1, 3)

#     def forward(self, x) :
#         # 先应用PConv
#         x = self.pconv(x)

#         # 计算各池化操作的权重
#         weights = self.adaptive_gate(x)

#         # 应用不同尺度的池化
#         p3 = self.pool3(x)
#         p5 = self.pool5(x)
#         p7 = self.pool7(x)

#         # 加权融合
#         return weights[:, 0 :1] * p3 + weights[:, 1 :2] * p5 + weights[:, 2 :3] * p7


# class EnhancedPSAWithSPPF(nn.Module) :
#     """改进的PSAWithSPPF模块，结合PConv思想"""

#     def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, n_div=4) :
#         super().__init__()
#         self.c = c
#         self.shortcut = shortcut
#         self.n_div = n_div

#         # 高效通道注意力
#         self.attn = EfficientChannelAttention(c)

#         # 使用PConv减少SPPF分支的输入通道数
#         self.sppf_pconv = PConv(c, c // 2, n_div=n_div)

#         # 多尺度自适应池化
#         self.adaptive_pool = AdaptiveMultiScalePooling(c // 2, n_div=n_div)

#         # 轻量级特征融合
#         self.sppf_conv_expand = DSConv2D(c // 2, c, 1, 1)

#         # 高效FFN，使用PConv
#         self.ffn = nn.Sequential(
#             PConv(c, c * 2, n_div=n_div),
#             nn.GELU(),
#             DSConv2D(c * 2, c, 1, act=False)
#         )

#         # 门控机制
#         self.gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, max(c // 4, 8), 1),
#             nn.ReLU(),
#             nn.Conv2d(max(c // 4, 8), 2, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x) :
#         residual = x

#         # 计算分支权重
#         gate_weights = self.gate(x)

#         # 注意力分支
#         attn_out = self.attn(x) * gate_weights[:, 0 :1]

#         # SPPF分支
#         sppf_out = self.sppf_pconv(x)
#         sppf_out = self.adaptive_pool(sppf_out)
#         sppf_out = self.sppf_conv_expand(sppf_out) * gate_weights[:, 1 :2]

#         # 分支融合
#         if self.shortcut :
#             x = residual + attn_out + sppf_out
#         else :
#             x = attn_out + sppf_out

#         # FFN处理
#         ffn_out = self.ffn(x)

#         if self.shortcut :
#             x = x + ffn_out
#         else :
#             x = ffn_out

#         return x


# class EnhancedC2PSA3(nn.Module) :
#     """改进的C2PSA3模块，结合PConv思想"""

#     def __init__(self, c1, c2, n=1, e=0.5, n_div=4) :
#         super().__init__()
#         assert c1 == c2
#         self.c = int(c1 * e)
#         self.n = n
#         self.n_div = n_div

#         # 使用PConv替代标准卷积
#         self.cv1 = PConv(c1, 2 * self.c, n_div=n_div)
#         self.cv2 = PConv(2 * self.c, c1, n_div=n_div)

#         # 动态路径选择
#         self.path_selector = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(self.c, max(self.c // 4, 8), 1),
#             nn.ReLU(),
#             nn.Conv2d(max(self.c // 4, 8), n, 1),
#             nn.Softmax(dim=1)
#         )

#         # 使用改进的PSAWithSPPF模块
#         self.m = nn.ModuleList([EnhancedPSAWithSPPF(
#             self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1), n_div=n_div
#         ) for _ in range(n)])

#     def forward(self, x) :
#         # 分割通道
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)

#         # 动态路径选择
#         if self.n > 1 :
#             path_weights = self.path_selector(b)
#             b_out = 0
#             for i, module in enumerate(self.m) :
#                 b_out = b_out + path_weights[:, i :i + 1] * module(b)
#         else :
#             b_out = self.m[0](b)

#         # 合并分支
#         return self.cv2(torch.cat((a, b_out), 1))

######################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLCA(nn.Module) :
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5) :
        super(MLCA, self).__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_weight = local_weight
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x) :
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c,
                                                                                                            self.local_size,
                                                                                                            self.local_size)
        y_global_transpose = y_global.transpose(-1, -2).unsqueeze(-1)

        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])

        return x * att_all




class Conv2d_BN(torch.nn.Sequential) :
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000) :
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse_self(self) :
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2 :], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module) :
    def __init__(self, m, drop=0.) :
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x) :
        if self.training and self.drop > 0 :
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else :
            return x + self.m(x)

    @torch.no_grad()
    def fuse_self(self) :
        if isinstance(self.m, Conv2d_BN) :
            m = self.m.fuse_self()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d) :
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else :
            return self


class SHSA_GroupNorm(torch.nn.GroupNorm) :
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs) :
        super().__init__(1, num_channels, **kwargs)


class SHSABlock_FFN(torch.nn.Module) :
    def __init__(self, ed, h) :
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.SiLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x) :
        x = self.pw2(self.act(self.pw1(x)))
        return x


class SHSA(torch.nn.Module) :
    """Single-Head Self-Attention"""

    def __init__(self, dim, qk_dim, pdim) :
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = SHSA_GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0))

    def forward(self, x) :
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))

        return x

import torch
import torch.nn as nn


class TridentBlock(nn.Module) :
    def __init__(self, c1, c2, stride=1, c=False, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False) :
        super(TridentBlock, self).__init__()
        self.stride = stride
        self.c = c
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias :
            self.bias = nn.Parameter(torch.Tensor(c2))
        else :
            self.bias = None

        if self.bias is not None :
            nn.init.constant_(self.bias, 0)

    def forward_for_small(self, x) :
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[0],
                                   dilation=self.dilate[0])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_middle(self, x) :
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[1],
                                   dilation=self.dilate[1])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_big(self, x) :
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[2],
                                   dilation=self.dilate[2])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward(self, x) :
        xm = x
        base_feat = []
        if self.c is not False :
            x1 = self.forward_for_small(x)
            x2 = self.forward_for_middle(x)
            x3 = self.forward_for_big(x)
        else :
            x1 = self.forward_for_small(xm[0])
            x2 = self.forward_for_middle(xm[1])
            x3 = self.forward_for_big(xm[2])

        base_feat.append(x1)
        base_feat.append(x2)
        base_feat.append(x3)

        return base_feat


class RFEM(nn.Module) :
    def __init__(self, c1, c2, n=1, e=0.5, stride=1) :
        super(RFEM, self).__init__()
        c = True
        layers = []
        layers.append(TridentBlock(c1, c2, stride=stride, c=c, e=e))
        c1 = c2
        for i in range(1, n) :
            layers.append(TridentBlock(c1, c2))
        self.layer = nn.Sequential(*layers)
        # self.cv = Conv(c2, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x) :
        out = self.layer(x)
        out = out[0] + out[1] + out[2] + x
        out = self.act(self.bn(out))
        return out


class Residual(nn.Module) :
    def __init__(self, fn) :
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x) :
        return self.fn(x) + x


class ConvMixer(nn.Module) :
    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=4, reduction=16) :
        super(ConvMixer, self).__init__()
        if c1 != c2 :
            c2 = c1
        self.DConvN = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(c2),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(c2, c2, kernel_size, groups=c2, padding=1),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(c2, c1, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(depth)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x) :
        b, c, _, _ = x.size()
        y = self.DConvN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)


class SEAM(nn.Module) :
    def __init__(self, c1, c2, n, reduction=16) :
        super(SEAM, self).__init__()
        if c1 != c2 :
            c2 = c1
        self.DCovN = nn.Sequential(
            # nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1),
            # nn.GELU(),
            # nn.BatchNorm2d(c2),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)

    def forward(self, x) :
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer) :
        if isinstance(layer, (nn.Conv2d, nn.Linear)) :
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None :
                torch.nn.init.constant_(layer.bias, 0)


def DcovN(c1, c2, depth, kernel_size=3, patch_size=3) :
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.SiLU(),
                nn.BatchNorm2d(c2)
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return dcovn


class MultiSEAM(nn.Module) :
    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=[3, 5, 7], reduction=16) :
        super(MultiSEAM, self).__init__()
        if c1 != c2 :
            c2 = c1
        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x) :
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)


class C3RFEM(nn.Module) :
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5) :  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        # self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.rfem = RFEM(c_, c_, n)
        self.m = nn.Sequential(*[RFEM(c_, c_, n=1, e=e) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x) :
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))





import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import math


def autopad(k, p=None, d=1) :  # kernel, padding, dilation
    if d > 1 :  # 如果采用扩张卷积,则计算扩张后实际的kernel大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None :
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DSConv(_ConvNd) :
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, groups=1, padding_mode='zeros', bias=False, block_size=32, KDSBias=False,
                 CDS=False) :
        padding = _pair(autopad(kernel_size, padding, dilation))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        blck_numb = math.ceil(((in_channels) / (block_size * groups)))
        super(DSConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # KDS weight From Paper
        self.intweight = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.alpha = torch.Tensor(out_channels, blck_numb, *kernel_size)

        # KDS bias From Paper
        self.KDSBias = KDSBias
        self.CDS = CDS

        if KDSBias :
            self.KDSb = torch.Tensor(out_channels, blck_numb, *kernel_size)
        if CDS :
            self.CDSw = torch.Tensor(out_channels)
            self.CDSb = torch.Tensor(out_channels)

        self.reset_parameters()

    def get_weight_res(self) :
        # Include expansion of alpha and multiplication with weights to include in the convolution layer here
        alpha_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Include KDSBias
        if self.KDSBias :
            KDSBias_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Handy definitions:
        nmb_blocks = self.alpha.shape[1]
        total_depth = self.weight.shape[1]
        bs = total_depth // nmb_blocks

        llb = total_depth - (nmb_blocks - 1) * bs

        # Casting the Alpha values as same tensor shape as weight
        for i in range(nmb_blocks) :
            length_blk = llb if i == nmb_blocks - 1 else bs

            shp = self.alpha.shape  # Notice this is the same shape for the bias as well
            to_repeat = self.alpha[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
            repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
            alpha_res[:, i * bs :(i * bs + length_blk), ...] = repeated.clone()

            if self.KDSBias :
                to_repeat = self.KDSb[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
                repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
                KDSBias_res[:, i * bs :(i * bs + length_blk), ...] = repeated.clone()

        if self.CDS :
            to_repeat = self.CDSw.view(-1, 1, 1, 1)
            repeated = to_repeat.expand_as(self.weight)
            print(repeated.shape)

        # Element-wise multiplication of alpha and weight
        weight_res = torch.mul(alpha_res, self.weight)
        if self.KDSBias :
            weight_res = torch.add(weight_res, KDSBias_res)
        return weight_res

    def forward(self, input) :
        # Get resulting weight
        # weight_res = self.get_weight_res()

        # Returning convolution
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation,
                        self.groups)


class DSConv2D(Conv) :
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True) :
        super().__init__(inc, ouc, k, s, p, g, d, act)
        self.conv = DSConv(inc, ouc, k, s, p, g, d)


class Bottleneck_DSConv(nn.Module) :
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5) :  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DSConv2D(c1, c_, 1, 1)
        self.cv2 = DSConv2D(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x) :
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_DSConv(nn.Module) :
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5) :
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = DSConv2D(c1, 2 * self.c, 1, 1)
        self.cv2 = DSConv2D((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DSConv(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x) :
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x) :
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


########################################

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


# class SPPELAN(nn.Module):
#     """SPP-ELAN."""

#     def __init__(self, c1, c2, c3, k=5):
#         """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)

#     def forward(self, x):
#         """Forward pass through SPPELAN layer."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         return self.cv5(torch.cat(y, 1))
########################

class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


###################

class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
#########

#######

class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))
class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
 
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)
 
    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))
class lightC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(improveBottleneckWithdwconvWithodconvwitham22(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class DyDCNv2(nn.Module) :
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)) :
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm :
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask) :
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm :
            x = self.norm(x)
        return x

###
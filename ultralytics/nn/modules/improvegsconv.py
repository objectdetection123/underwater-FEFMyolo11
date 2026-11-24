import torch
from torch import nn


def autopad(k, p=None) :  # kernel, padding
    # Pad to 'same'
    if p is None :
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module) :
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True) :  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x) :
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x) :
        return self.act(self.conv(x))





import torch
import torch.nn as nn
from einops import rearrange
from .block import PSABlock, C2PSA


class CAFM(nn.Module) :
    def __init__(self, dim, num_heads=8, bias=False) :
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x) :
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output

import torch
import torch.nn as nn


class GhostConv(nn.Module) :
    """Ghost Convolution 替代 DWConv"""

    def __init__(self, c1, c2, k=5, s=1, g=1, act=True) :
        super().__init__()
        # 主卷积分支生成c2//2通道
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, c2 // 2, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU() if act else nn.Identity()
        )
        # 廉价操作分支生成剩余通道
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(c2 // 2, c2 // 2, k, 1, k // 2, groups=c2 // 2, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x) :
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)  # 沿通道维度拼接


class improvegsconv(nn.Module) :
    """改进后的 GSConv：GhostConv 后接并行 CAFM 和 DWConv 分支"""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True) :
        super().__init__()
        # Ghost卷积核（主分支）
        self.ghost_conv = GhostConv(c1, c2 // 2, k, s, g, act)

        # 新增CAFM注意力分支
        self.cafm_branch = nn.Sequential(
            CAFM(c2 // 2),  # 通道注意力模块
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU() if act else nn.Identity()
        )

        # 原有的DW卷积分支（保持原结构）
        self.dw_branch = nn.Sequential(
            nn.Conv2d(c2 // 2, c2 // 2, 5, 1, 2, groups=c2 // 2, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU() if act else nn.Identity()
        )

        # 最终激活函数
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x) :
        # 主路径：Ghost卷积
        x_base = self.ghost_conv(x)

        # 并行分支处理
        x_cafm = self.cafm_branch(x_base)  # CAFM注意力分支
        x_dw = self.dw_branch(x_base)  # DW卷积分支

        # 特征融合
        x_fused = torch.cat([x_cafm, x_dw], dim=1)

        # 通道混洗（保持原逻辑）
        b, c, h, w = x_fused.size()
        x_shuffled = x_fused.view(b, 2, c // 2, h, w)
        x_shuffled = x_shuffled.permute(0, 2, 1, 3, 4).contiguous()
        x_shuffled = x_shuffled.view(b, c, h, w)

        return self.act(x_shuffled)
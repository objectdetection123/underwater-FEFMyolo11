import torch
import torch.nn as nn

#
# class EfficientLocalizationAttention(nn.Module) :
#     def __init__(self, channel, kernel_size=7) :
#         super(EfficientLocalizationAttention, self).__init__()
#         self.pad = kernel_size // 2
#         self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
#         self.gn = nn.GroupNorm(16, channel)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x) :
#         b, c, h, w = x.size()
#
#         # 处理高度维度
#         x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
#         x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
#
#         # 处理宽度维度
#         x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
#         x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)
#
#
#         return x * x_h * x_w
#
#
# # 示例用法 ELABase(ELA-B)
# # if __name__ == "__main__" :
# #     # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
# #     dummy_input = torch.randn(2, 64, 32, 32)
# #
# #     # 初始化模块
# #     ela = EfficientLocalizationAttention(channel=dummy_input.size(1), kernel_size=7)
# #
# #     # 前向传播
# #     output = ela(dummy_input)
# #     # 打印出输出张量的形状，它将与输入形状相匹配。
# #     print(f"输出形状: {output.shape}")
#
# # """
# # 为了在考虑参数数量的同时优化ELA的性能，作者引入了四种方案: ELA-Tiny(ELA-T)，ELABase(ELA-B)，ELA-Smal(ELA-S)和ELA-Large(ELA-L)。
# # 1.ELA-T的参数配置定义为 kernel size=5,groups =in channels， num group=32:
# # 2.ELA-B的参数配置定义为 kernel size=7，groups =in_channels， num_group =16:
# # 3.ELA-S的参数配置为 kernel size=5,groups=in_channels/8, num_group=16。
# # 4.ELA-L的参数配置为 kernel_size=7,groups=in _channels /8，num_group=16 。
# # """
#
# # ELA (coordinate attention)
#
#
#
# class ELA(nn.Layer) :
#     def __init__(self, in_ch, kernel_size=7) :
#         super(ELA, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2D((1, None))
#
#         pad = kernel_size // 2
#         self.conv1 = nn.Conv1D(in_ch, in_ch, kernel_size=kernel_size, padding=pad, groups=in_ch, bias_attr=False)
#         self.gn = nn.GroupNorm(16, in_ch)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x) :
#         identity = x
#         b, c, h, w = x.shape
#
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x)
#
#         x_h = x_h.reshape([b, c, h])
#         x_w = x_w.reshape([b, c, w])
#
#         x_h = self.conv1(x_h)
#         x_h = self.gn(x_h)
#         x_h = self.sigmoid(x_h).reshape([b, c, h, 1])
#
#         x_w = self.conv1(x_w)
#         x_w = self.gn(x_w)
#         x_w = self.sigmoid(x_w).reshape([b, c, 1, w])
#
#         out = identity * x_w * x_h
#
#         return out
#
# #
# # if __name__ == '__main__' :
# #     ela = ELA(512)  # in_channel
# #     x = paddle.randn([64, 512, 14, 14])  # (batchsize, channel, H, W)
# #     y = ela(x)
# #     print(y.shape)
import torch
import torch.nn as nn


class ELA(nn.Module) :
    def __init__(self, in_channels, phi='B') :
        super(ELA, self).__init__()
        '''
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        '''
        Kernel_size = {'T' : 5, 'B' : 7, 'S' : 5, 'L' : 7}[phi]
        groups = {'T' : in_channels, 'B' : in_channels, 'S' : in_channels // 8, 'L' : in_channels // 8}[phi]
        num_groups = {'T' : 32, 'B' : 16, 'S' : 16, 'L' : 16}[phi]
        pad = Kernel_size // 2
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input) :
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        x_h = self.con1(x_h)  # [b,c,h]
        x_w = self.con1(x_w)  # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        return x_h * x_w * input


if __name__ == "__main__" :
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(2, 256, 40, 40)
    ela = ELA(in_channels=256, phi='T')
    output = ela(input)
    print(output.size())
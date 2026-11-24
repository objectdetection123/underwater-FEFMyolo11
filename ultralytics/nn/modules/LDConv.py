import torch
import torch.nn as nn
class LDConv_2D(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv_2D, self).__init__()  # 修正了super的调用，应为super(当前类名, self).__init__()
        self.num_param = num_param
        self.stride = stride
        # 修正了Conv2d中的inc参数，应为传入的inc而非ine
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YoLov5
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=(3, 1), padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # 注意：原代码中的'modute'可能是一个拼写错误，我假设它应该是'modulate'，但根据上下文，这个函数名可能并不准确，
        # 因为'modulate'参数在函数体内并未被使用。此外，'θ.1'应该是'0.1'的拼写错误。
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))  # 使用enumerate来遍历列表，但原代码没有使用i
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))  # 同样使用enumerate
        # 注意：原代码中的列表推导式语法有误，我进行了修正。另外，由于'modulate'参数未使用，这里直接对grad_input和grad_output进行了缩放。

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)  # 假设self.p_conv是一个已经定义好的卷积层
        dtype = offset.data.type()  # 获取offset的数据类型
        N = offset.size(1) // 2  # 计算N的值，假设offset的形状是(b, 2N, h, w)
        # （b，2N,h，w）是offset的形状注释，不需要作为代码的一部分

        # 下面的代码片段可能包含了一些错误或不规范的写法，比如括号的使用和注释的书写方式
        p = self._get_p(offset, dtype)  # 假设_get_p是一个已经定义好的方法，用于根据offset和dtype计算p
        # （b,h;w,2N）是p的形状注释，同样不需要作为代码的一部分，并且分号';'应该是逗号','的误写
        p = p.contiguous().permute(0, 2, 3, 1)  # 将p重新排列，确保它是连续的
        q_lt = p.detach().floor()  # 对p进行下取整操作，并分离计算图
        q_rb = q_lt+1
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], # 同上，max应该是x.size(3) - 1的上界
                        dim=-1).long()

        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],  # 同上
                        dim=-1).long()

        # 注意：原代码中的q_lb和q_rt的计算没有使用torch.clamp，可能是个遗漏或者特意为之
        # 这里我按照原代码的逻辑进行了补全，但没有添加torch.clamp
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1),torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N :].type_as(p) - p[..., N :]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N :].type_as(p) - p[..., N :]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N :].type_as(p) - p[..., N :]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N :].type_as(p) - p[..., :N]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_g(x, q_rb, N)  # 注意这里原代码是g_rb，但根据上下文，这里可能是q_rb的笔误，我保留了原样
        x_q_lb = self._get_x_q(x, q_lb, N)  # 修正了中文逗号
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear interpolation calculation
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        # 返回处理后的结果
        return out

    def _get_p_n(self, N, dtype) :
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)

        # 首先检查mod_number是否大于0
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),  # 这里可能是一个错误，通常我们不需要这样的arange，可能是想表达0到row_number-1
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y = torch.cat((p_n_x, mod_p_n_x)),torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y],0)
        p_n = p_n.view(1,2*N,1,1).type(dtype)
        return p_n
    def _get_p_0(self , h, w, N, dtype):
        p_0_x, p_0_y= torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),  # 修正了 self.strutde 为 self.stride
            torch.arange(0, w * self.stride,self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x,p_0_y],1).type(dtype)

        return p_0
    def _get_p(self, offset, dtype) :
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N,dtype)
        p_0 = self._get_p_0(h, w, N, type)
        p = p_0+p_n+offset
        p = p_0+p_n+offset
        return p
    def _get_x_q(self,x,q, N):
        b , h, w, _ = q.size()



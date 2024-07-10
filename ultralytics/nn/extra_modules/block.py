import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from einops import rearrange
from collections import OrderedDict
from typing import Optional, Dict


from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *
from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible

from timm.layers import DropPath

from torch.nn.common_types import _size_2_t
from collections import OrderedDict
from scipy.stats import special_ortho_group, unitary_group

__all__ = ["BasicBlock_Ortho_Shuffle"]


def sample_orthogonal_matrix(n, group_type='SO'):
    """在特殊正交群SO(n)或特殊幺正群SU(n)上随机采样正交矩阵"""
    if group_type == 'SO':
        return torch.from_numpy(special_ortho_group.rvs(n)).float()
    elif group_type == 'SU':
        return torch.from_numpy(unitary_group.rvs(n)).float()
    else:
        raise ValueError(f"未知的群类型:{group_type}")

def initialize_orthogonal_filters(c, h, w, group_type='SO'):
    """采样正交滤波器权重"""
    n = max(h, w)  
    filters = []
    for _ in range(c):
        m = sample_orthogonal_matrix(n, group_type)
        filters.append(m[:h,:w].unsqueeze(0).unsqueeze(0)) 
    return torch.cat(filters)

class GramSchmidtTransform(torch.nn.Module):
    instance = {}
    constant_filter: torch.Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int, group_type='SO'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h, group_type).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())
        
    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)

class Attention_Ortho(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: GramSchmidtTransform, input: torch.Tensor):
        while input[0].size(-1) > 1:
            input = FWT(input)
        b = input.size(0)
        return input.view(b, -1)


class BasicBlock_Ortho_Shuffle(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', height=64, variant='d', group_type='SO'):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 
        
        self._excitation = nn.Sequential(
            nn.Linear(in_features=ch_out, out_features=round(ch_out / 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(ch_out / 16), out_features=ch_out, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention_Ortho()
        self.F_C_A = GramSchmidtTransform.build(ch_out, height)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0),out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        out = excitation * out 
        
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.channel_shuffle(out, 2)
        out = self.act(out)

        return out
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x







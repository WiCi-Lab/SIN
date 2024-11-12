# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:29:40 2021

@author: 5106
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:27:27 2021

@author: 5106
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

# MLP module
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=False , drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#https://zhuanlan.zhihu.com/p/569674523    
class CycleFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        这里的kernel_size实际使用的时候时3x1或者1x3
        """
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        # 被偏移调整的1x1卷积的权重，由于后面使用torchvision提供的可变形卷积的函数，所以权重需要自己构造
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))
        # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # 要注意，这里是在注册一个buffer，是一个常量，不可学习，但是可以保存到模型权重中。
        self.register_buffer('offset', self.gen_offset())

    def gen_offset(self):
        """
        生成卷积核偏移量的核心操作。
        要想理解这一函数的操作，需要首先理解后面使用的deform_conv2d_tv的具体用法。
        具体可见：https://pytorch.org/vision/0.10/ops.html#torchvision.ops.deform_conv2d
        这里对于offset参数的要求是：
        offset (Tensor[batch_size,
                       2 * offset_groups * kernel_height * kernel_width,
                       out_height,
                       out_width])
                       – offsets to be applied for each position in the convolution kernel.
        也就是说，对于样本s的输出特征图的通道c中的位置(x,y)，这个函数会从offset中取出，形状为
        kernel_height*kernel_width的卷积核所对应的偏移参数为
        offset[s, 0:2*offset_groups*kernel_height*kernel_width, x, y]
        也就是这一系列参数都是对应样本s的单个位置(x,y)的。
        针对不同的位置可以有不同的offset，也可以有相同的（下面的实现就是后者）。
        对于这2*offset_groups*kernel_height*kernel_width个数，涉及到对于输入特征通道的分组。
        将其分成offset_groups组，每份单独拥有一组对应于卷积核中心位置的相对偏移量，
        共2*kernel_height*kernel_width个数。
        对于每个核参数，使用两个量来描述偏移，即h方向和w方向相对中心位置的偏移，
        即下面代码中的减去kernel_height//2或者kernel_width//2。
        需要注意的是，当偏移位置位于padding后的tensor边界外，则是将网格使用0补齐。
        如果网格上有边界值，则使用边界值和用0补齐的网格顶点来计算双线性插值的结果。
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                # 这里计算了一个相对偏移位置。
                # deform_conv2d使用的以对应输出位置为中心的偏移坐标索引方式
                offset[0, 2 * i + 1, 0, 0] = (
                    (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
                )
            else:
                offset[0, 2 * i + 0, 0, 0] = (
                    (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                )
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input,
                                self.offset.expand(B, -1, H, W),
                                self.weight,
                                self.bias,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
    
class CycleMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CycleBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

    
def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x


def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=3):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) #bs,k,n,c
        a=torch.sum(torch.sum(x_all,1),1) #bs,c
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) #bs,kc
        hat_a=hat_a.reshape(b,self.k,c) #bs,k,c
        bar_a=self.softmax(hat_a) #bs,k,c
        attention=bar_a.unsqueeze(-2) # #bs,k,1,c
        out=attention*x_all # #bs,k,n,c
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512 ):
        super().__init__()
        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention(channel=channels)

    def forward(self, x):
        b,h,w,c = x.size()

        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all=torch.stack([x1,x2,x3],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        # x=x.permute(0,3,1,2)
        return x


# MLP-based Permutator module
class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim1, dim2, dim3, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim3, dim3, bias=qkv_bias)

        self.reweight = Mlp(dim1, dim1 // 2, dim1 *3)

        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        seg = dim1//segment_dim
        
        self.complex_weight = nn.Parameter(torch.randn( dim2//seg, dim3//seg//2+1, dim1, 2, dtype=torch.float32) * 0.02)
        
        
        self.fuse=nn.Linear(2*dim1,dim1)
        
        self.s2att = S2Attention(channels=dim1)


    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H*S)
        
        # xf = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # weight = torch.view_as_complex(self.complex_weight)
        # xf = xf * weight
        # xf = torch.fft.irfft2(xf, s=(H, W), dim=(1, 2), norm='ortho')
        # xf = self.s2att(x)
        
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
        

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W*S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        
        # x_fuse=torch.cat([x,xf],dim=3)
        # x=self.fuse(x_fuse)

        # x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
# MLP-based Permutator module
class WeightedPermuteMLP1(nn.Module):
    def __init__(self, dim1, dim2, dim3, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c1 = nn.Linear(dim1//4, dim1, bias=qkv_bias)
        self.mlp_c2 = nn.Linear(dim1//4, dim1, bias=qkv_bias)
        self.mlp_c3 = nn.Linear(dim1, dim1, bias=qkv_bias)
        
        self.mlp_h = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim3, dim3, bias=qkv_bias)

        self.reweight = Mlp(dim1, dim1 // 2, dim1 *3)

        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        seg = dim1//segment_dim
        
        self.complex_weight = nn.Parameter(torch.randn( dim2//seg, dim3//seg//2+1, dim1, 2, dtype=torch.float32) * 0.02)
        
        
        self.fuse = nn.Linear(3*dim1, dim1)


    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H*S)
        
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
        

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W*S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        # c1 = self.mlp_c1(x[:,:,:,0:C//4])
        # c2 = self.mlp_c2(x[:,:,:,::4])
        c3 = self.mlp_c3(x)
        
        # a = (c1 + c2 + c3).permute(0, 3, 1, 2).flatten(2).mean(2)
        # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        # c = c1 * a[0] + c2 * a[1] + c3 * a[2]
        
        # x_fuse=torch.cat([c1,c2,c3],dim=3)
        # c=self.fuse(x_fuse)

        # a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        # x = h * a[0] + w * a[1] + c * a[2]
        
        x_fuse=torch.cat([h,w,c3],dim=3)
        x=self.fuse(x_fuse)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
# MLP-based Permutator module
class FirstMLP(nn.Module):
    def __init__(self, in_ch, dim1):
        super().__init__()

        self.mlp_c1 = nn.Linear(in_ch//2, dim1)
        self.mlp_c2 = nn.Linear(in_ch//2, dim1)
        self.mlp_c3 = nn.Linear(in_ch, dim1)
        self.reweight = Mlp(dim1, dim1 // 2, dim1 *3)

        self.fuse = nn.Linear(4*dim1, dim1)


    def forward(self, x):
        x1 = x
        x= x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        c1 = self.mlp_c1(x[:,:,:,0:C//2])
        c2 = self.mlp_c2(x[:,:,:,::2])
        c3 = self.mlp_c3(x)
        
        B, H, W, C = c3.shape
        
        a = (c1 + c2 + c3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        c = c1 * a[0] + c2 * a[1] + c3 * a[2]
        
        x_fuse=torch.cat([c,c1,c2,c3],dim=3)
        x=self.fuse(x_fuse)
        x= x.permute(0, 3, 1, 2)
        # x = x1 + x
        
        return x
    
# Complete Permutator block
class PermutatorBlock1(nn.Module):

    def __init__(self, dim1, dim2, dim3, segment_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=False, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn = WeightedPermuteMLP1):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.attn = mlp_fn(dim1, dim2, dim3, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        # self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam
        
        seg = dim1//segment_dim
        
        # self.complex_weight = nn.Parameter(torch.randn( dim2//seg, dim3//seg//2+1, dim1, 2, dtype=torch.float32) * 0.02)
        
        # self.mlp_c = nn.Linear(dim1, dim1, bias=qkv_bias)
        
        self.MLP_temporal = MLP_temporal(dim1//2+1)
        

    def forward(self, x):
        B, H, W, C = x.shape

        x = x + self.attn(self.norm1(x)) / self.skip_lam
        bias = x
        x = self.MLP_temporal(x, B, H, W) + bias
        
        # x = x + self.mlp(self.norm2(x)) / self.skip_lam
        
        # xf = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # x = xf * weight
        # x = self.mlp(self.norm2(x)) / self.skip_lam
        # x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho') + x
        # x = self.mlp_c(x)
        
        return x
    
# frequency-domain MLPs
# dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
# rb: the real part of bias, ib: the imaginary part of bias
class FreMLP(nn.Module):
    def __init__(self, embed_size):
        super(FreMLP, self).__init__()
        
        self.embed_size = embed_size
        self.sparsity_threshold = 0.01
        
        
    def forward(self, B, nd, dimension, x, r, i, rb, ib):
        
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
    
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )
    
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )
    
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y
    
# frequency temporal learner
class MLP_temporal(nn.Module):
    def __init__(self, embed_size):
        super(MLP_temporal, self).__init__()
        
        self.scale = 0.02
        self.embed_size = embed_size
    
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        
        self.FreMLP = FreMLP(self.embed_size)

    def forward(self, x, B, N, L):
        # [B, N, T, D]
        
        B, H, W, C = x.shape
        
        x = torch.fft.rfft(x, dim=3, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=C, dim=3, norm="ortho")
        return x

# Complete Permutator block
class PermutatorBlock(nn.Module):

    def __init__(self, dim1, dim2, dim3, segment_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=False, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn = WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.attn = mlp_fn(dim1, dim2, dim3, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam
        
        seg = dim1//segment_dim
        
        self.complex_weight = nn.Parameter(torch.randn( dim2//seg, dim3//seg//2+1, dim1, 2, dtype=torch.float32) * 0.02)
        
        self.mlp_c = nn.Linear(dim1, dim1, bias=qkv_bias)
        
        self.MLP_temporal = MLP_temporal(dim1//2+1)
        

    def forward(self, x):
        B, H, W, C = x.shape

        x = x + self.attn(self.norm1(x)) / self.skip_lam
        bias = x
        x = self.MLP_temporal(x, B, H, W) + bias
        
        # x = x + self.mlp(self.norm2(x)) / self.skip_lam
        
        # xf = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # x = xf * weight
        # x = self.mlp(self.norm2(x)) / self.skip_lam
        # x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho') + x
        # x = self.mlp_c(x)
        
        return x

# Convolutional module
class conv_block1(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, strides,pads, dilas):
        super(conv_block1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=strides, padding=pads, dilation=dilas,bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.3)
            )

    def forward(self, x):

        x = self.conv(x)
        return x

# Upsampling module
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.3)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
# Residual module
class _Res_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.res_conb = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.instan = nn.InstanceNorm2d(out_ch)

    def forward(self, x,al=1):

        y = self.relu(self.instan(self.res_conv(x)))
        y = self.res_conb(y)
        y *= al
        y = torch.add(y, x)
        return y

# frequency channel learner
# def MLP_channel(self, x, B, N, L):
#     # [B, N, T, D]
    
#     self.scale = 0.02
#     self.embed_size = x.shape[3]
#     self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#     self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#     self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#     self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
    
#     x = x.permute(0, 2, 1, 3)
#     # [B, T, N, D]
#     x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
#     y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
#     x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
#     x = x.permute(0, 2, 1, 3)
#     # [B, N, T, D]
#     return x


# def forward(self, x):
#     # x: [Batch, Input length, Channel]
#     B, T, N = x.shape
#     # embedding x: [B, N, T, D]
#     x = self.tokenEmb(x)
#     bias = x
#     # [B, N, T, D]
#     if self.channel_independence == '1':
#         x = self.MLP_channel(x, B, N, T)
#     # [B, N, T, D]
#     x = self.MLP_temporal(x, B, N, T)
#     x = x + bias
#     x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
#     return x

class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,mode='fc'):
        super().__init__()
        
        
        self.fc_h = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)  
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.mode=mode
        
        if mode=='fc':
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())  
        else:
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU()) 
                    


    def forward(self, x):
     
        B, C, H, W = x.shape
        theta_h=self.theta_h_conv(x)
        theta_w=self.theta_w_conv(x)

        x_h=self.fc_h(x)
        x_w=self.fc_w(x)      
        x_h=torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)],dim=1)
        x_w=torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)],dim=1)

#         x_1=self.fc_h(x)
#         x_2=self.fc_w(x)
#         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
#         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)
        
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c,output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x
        
class WaveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

# Channel prediction network
class channel_pre(nn.Module):
    def __init__(self):
        super(channel_pre, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        in_ch=5
        out_ch=5     
        
        self.inter = nn.Upsample(size=None, scale_factor=(4,1), mode='bicubic', align_corners=None)

                
        self.Conv11 = conv_block1(in_ch, filters[0],1,1,1)
        self.Conv22 = conv_block1(filters[0], filters[1],1,1,1)
        self.Conv33 = conv_block1(filters[1], filters[2],1,1,1)

        self.Conv = nn.Conv2d(filters[2], out_ch, kernel_size=1, stride=1, padding=0)
        
        seg_dim1 = 16
        seg1 = filters[1]//seg_dim1
        
        seg_dim33 = 16
        seg33 = filters[2]//seg_dim33
        
        seg_dim2 = 16
        seg2 = filters[0]//seg_dim2

        self.mlp_mixer11 = PermutatorBlock(dim1 = filters[0], dim2 = 64*seg2, dim3 = 72*seg2, segment_dim=seg_dim2)
        self.mlp_mixer22 = PermutatorBlock(dim1 = filters[1], dim2 = 64*seg1, dim3 = 72*seg1, segment_dim=seg_dim1)
        self.mlp_mixer33 = PermutatorBlock(dim1 = filters[2], dim2 = 64*seg33, dim3 = 72*seg33, segment_dim=seg_dim33)

        self.MLP_temporal = MLP_temporal(in_ch//2+1)

    def forward(self, x):
        
        x = self.MLP_temporal(x)
        
        e1 = self.Conv11(x) 
        e1 = rearrange(e1, 'b c h w -> b h w c')
        e1 = self.mlp_mixer11(e1)
        e1 = rearrange(e1, 'b h w c-> b c h w')

        
        e2 = self.Conv22(e1)
        e2 = rearrange(e2, 'b c h w -> b h w c')
        e2 = self.mlp_mixer22(e2)
        e2 = rearrange(e2, 'b h w c-> b c h w')
        
        
        e3 = self.Conv33(e2)
        e3 = rearrange(e3, 'b c h w -> b h w c')
        e3 = self.mlp_mixer33(e3)
        e3 = rearrange(e3, 'b h w c-> b c h w')
        
        out = self.Conv(e3)
            
        return out

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

# Channel estimation network
class channel_est(nn.Module):
    def __init__(self, in_ch, out_ch, Hn,Hw):
        super(channel_est, self).__init__()

        n1 = 48
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # in_ch=8
        # out_ch=8     
        
        self.inter = nn.Upsample(size=None, scale_factor=(4,1), mode='bicubic', align_corners=None)
                
        self.Conv11 = conv_block1(in_ch, filters[0],1,2,2)
        # self.Conv11 = FirstMLP(in_ch, filters[0])
        # self.Conv1T = FirstMLP(in_ch, filters[0])

        self.Conv22 = conv_block1(filters[0], filters[1],2,1,1)
        self.Conv33 = conv_block1(filters[1], filters[2],2,1,1)
        
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up33 = conv_block1(filters[2], filters[1],1,1,1)       

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up22 = conv_block1(filters[1], filters[0],1,1,1)
        
        
        self.Conv = nn.Conv2d(filters[0], filters[0], kernel_size=1, stride=1, padding=0)
        
        seg_dim1 = 24
        seg1 = filters[1]//seg_dim1
        
        seg_dim33 = 24
        seg33 = filters[2]//seg_dim33
        
        seg_dim2 = 8
        seg2 = filters[0]//seg_dim2

        self.mlp_mixer11 = PermutatorBlock(dim1 = filters[0], dim2 = Hn*seg2, dim3 = Hw*seg2, segment_dim=seg_dim2)
        self.mlp_mixer22 = PermutatorBlock(dim1 = filters[1], dim2 = Hn//2*seg1, dim3 = Hw//2*seg1, segment_dim=seg_dim1)
        self.mlp_mixer33 = PermutatorBlock(dim1 = filters[2], dim2 = Hn//4*seg33, dim3 = Hw//4*seg33, segment_dim=seg_dim33)
        
        seg_dim2 = filters[0]//4
        seg2 = filters[0]//seg_dim2
        self.mlp_mixerT = PermutatorBlock(dim1 = filters[0], dim2 = Hn*seg2, dim3 = Hw*seg2, segment_dim=seg_dim2)
        self.ConvT = nn.Conv2d(filters[0], out_ch, kernel_size = 1, stride=1, padding=0)
        
        self.MLP_temporal = MLP_temporal(in_ch//2+1)

        self.weight1 = Scale(1)
    def forward(self, x):
        
        # x1 = rearrange(x, 'b c h w -> b h w c')
        # B, H, W, C = x1.shape
        # x1 = self.MLP_temporal(x1,B, H, W)
        # x1 = rearrange(x1, 'b h w c-> b c h w')
        # x = x+x1
        
        e1 = self.Conv11(x) 
        eT = e1
        e1 = rearrange(e1, 'b c h w -> b h w c')
        e1 = self.mlp_mixer11(e1)
        e1 = rearrange(e1, 'b h w c-> b c h w')

        
        e2 = self.Conv22(e1)
        e2 = rearrange(e2, 'b c h w -> b h w c')
        e2 = self.mlp_mixer22(e2)
        e2 = rearrange(e2, 'b h w c-> b c h w')
        
        
        e3 = self.Conv33(e2)
        e3 = rearrange(e3, 'b c h w -> b h w c')
        e3 = self.mlp_mixer33(e3)
        e3 = rearrange(e3, 'b h w c-> b c h w')
        

        d3 = self.Up3(e3) 
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up33(d3)
        d3 = rearrange(d3, 'b c h w -> b h w c')
        d3 = self.mlp_mixer22(d3)
        d3 = rearrange(d3, 'b h w c-> b c h w')
        

        d2 = self.Up2(d3) 
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up22(d2)
        d2 = rearrange(d2, 'b c h w -> b h w c')
        d2 = self.mlp_mixerT(d2)
        d2 = rearrange(d2, 'b h w c-> b c h w')
        
        # d2 = self.ConvT(d2)
        # d2 = self.Conv1T(d2)+eT
        # d2 = rearrange(d2, 'b c h w -> b h w c')
        # d2 = self.mlp_mixerT(d2)
        # d2 = rearrange(d2, 'b h w c-> b c h w')
        out = self.ConvT(d2)
            
        return out

# NMSE function
def NMSE_cuda(x, x_hat):
    x = x.contiguous().view(len(x), -1)
    x_hat = x_hat.contiguous().view(len(x_hat), -1)
    mse = torch.sum((x- x_hat) ** 2, axis=1)
    nmse = mse/torch.sum(x**2, axis=1)
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse
    
# def NMSE_cuda(x, x_hat):
#     x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
#     x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
#     x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
#     x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
#     power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
#     mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
#     nmse = mse / power
#     return nmse
 
 
# class NMSELoss(nn.Module):
#     def __init__(self, reduction='sum'):
#         super(NMSELoss, self).__init__()
#         self.reduction = reduction
 
#     def forward(self, x_hat, x):
#         nmse = NMSE_cuda(x, x_hat)
#         if self.reduction == 'mean':
#             nmse = torch.mean(nmse)
#         else:
#             nmse = torch.sum(nmse)
#         return nmse

# Data argumentation operations to avoid the network overfitting 
def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = int(h*cut_ratio), int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }

def cutmixup(
    im1, im2,    
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2

def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(2)

    im1 = im1[:,perm,:,:]
    im2 = im2[:,perm,:,:]

    return im1, im2

def rgb1(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2
    
    se = np.zeros(2)
    se[0]=1
    se[1]=-1
    
    r = np.random.randint(2)
    phase = se[r]
    im1[:,0,:,:] = phase*im1[:,0,:,:]
    im2[:,0,:,:] = phase*im2[:,0,:,:]
    r = np.random.randint(2)
    phase = se[r]
    im1[:,1,:,:] = phase*im1[:,1,:,:]
    im2[:,1,:,:] = phase*im2[:,1,:,:]

    return im1, im2

def cutmix(im1, im2, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2

def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    
    return im1, im2
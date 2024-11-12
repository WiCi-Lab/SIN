# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 10:35:07 2024

@author: WiCi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# Convolutional module
class conv_block1(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, strides,pads, dilas):
        super(conv_block1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=strides, padding=pads, dilation=dilas,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.3)
            )

    def forward(self, x):

        x = self.conv(x)
        return x

class Linear(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, seq_len,pred_len,enc_in,individual):
        super(Linear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len,pred_len,enc_in,individual):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len,pred_len,enc_in,individual):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]

from torch.nn import init    
    
class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv2d(in_channels,c_m,1)
        self.convB=nn.Conv2d(in_channels,c_n,1)
        self.convV=nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w
        B=self.convB(x) #b,c_n,h,w
        V=self.convV(x) #b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)

        return tmpZ 

# A2Attention network
class A2Attention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(A2Attention, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # in_ch=8
        # out_ch=2
                
        self.Conv11 = conv_block1(in_ch, filters[0],1,2,2)
        self.Conv22 = conv_block1(filters[0], filters[1],1,1,1)
    

        self.DoubleAttention1 = DoubleAttention(filters[0],128,128,True)
        
        self.DoubleAttention2 = DoubleAttention(filters[1],128,128,True)
        self.Conv = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        
        e1 = self.Conv11(x) 
        e1 = self.DoubleAttention1(e1)
        e1 = self.Conv22(e1) 
        e1 = self.DoubleAttention2(e1)
        
        out = self.Conv(e1)
            
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
 
from torch.nn.utils import weight_norm
 
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(n_outputs)
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.bn2 = nn.BatchNorm1d(n_outputs)
 
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
 
        self.net = nn.Sequential(self.conv1, self.bn1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.init_weights()
 
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
 
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
 
 
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, outputs, pre_len, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.pre_len = pre_len
 
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], outputs)
    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x[:, -self.pre_len:, :]
    
class TemporalConvNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_inputs, outputs, pre_len, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        self.pre_len = pre_len
 
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], outputs)
        
        numf = 64 
        self.conv0 = nn.Conv2d(in_ch, numf, (3, 3), (1, 1), padding=1)

        self.FeatureExtraction1 = FeatureExtraction(out_ch, numf)
        self.FeatureExtraction2 = FeatureExtraction(out_ch, numf)

        # self.ImageReconstruction1 = ImageReconstruction(in_ch, out_ch)
        self.conv1 = nn.Conv2d(numf, out_ch, (3, 3), (1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(numf)
        self.bn2 = nn.BatchNorm2d(numf)
        
    def forward(self, LR):
        
        BATCH_SIZE, seq_len, H, W = LR.shape
        
        
        
        # convt_F1 = self.FeatureExtraction1(LR1)
        LR1 = LR.flatten(2)
        
        LR1 = LR1.permute(0, 2, 1)
        LR1 = self.network(LR1)
        LR1 = LR1.permute(0, 2, 1)
        LR1 = self.linear(LR1)
        
        LR1 = rearrange(LR1, 'b c (h w) -> b c h w', h= H)
        
        LR = LR1 + LR
        
        LR1 = self.conv0(LR)
        LR1 = self.bn1(LR1) 
        LR2 = LR1
        LR1 = self.FeatureExtraction1(LR1)
        LR1 = self.bn2(LR1) + LR2 
        LR1 = self.FeatureExtraction2(LR1)        
        HR_2 = self.conv1(LR1)
        
        # x = x.permute(0, 2, 1)
        # x = self.network(x)
        # x = x.permute(0, 2, 1)
        # x = self.linear(x)
        return HR_2
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=True),
            # nn.LeakyReLU(negative_slope=0.3),
            # nn.Linear(channel // reduction, channel, bias=True),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, x):
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class _Res_Blocka(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Res_Blocka, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res_conv = weight_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.res_conb = weight_norm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.ca = SELayer(out_ch)

    def forward(self, x,al=1):

        y = self.relu(self.bn1(self.res_conv(x)))
        # y = self.relu(self.res_conb(y))
        # y = self.ca(y)
        # y *= al
        y = torch.add(y, x)
        return y
    
class FeatureExtraction(nn.Module):
    def __init__(self, in_ch, numf):
        super(FeatureExtraction, self).__init__()
        # numf=64
        self.conv4 = weight_norm(nn.Conv2d(numf, numf, (3, 3), (1, 1), (1, 1)))

        # self.convt_F = nn.Upsample(size=None, scale_factor=(1,2), mode='nearest', align_corners=None)

        self.LReLus = nn.LeakyReLU(negative_slope=0.2)
        
        m_body = [
            _Res_Blocka(numf,numf) for _ in range(2)
        ]
        
        self.bn1 = nn.BatchNorm2d(numf)
        
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
            
        out = self.bn1(self.body(x))
        out =  self.LReLus(self.conv4(out))

        return out

class ImageReconstruction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ImageReconstruction, self).__init__()
        self.conv_R = weight_norm(nn.Conv2d(64, out_ch, (3, 3), (1, 1), padding=1))
        # self.convt_I = nn.Upsample(size=None, scale_factor=(1,2), mode='nearest', align_corners=None)
        self.conv_1 = nn.Conv2d(in_ch, out_ch, (3, 3), (1, 1), padding=1)
        
    def forward(self, LR, convt_F):
        convt_I = self.conv_1(LR)
        
        conv_R = self.conv_R(convt_F)
        
        HR = convt_I + conv_R
        
        return HR
        
        
class LPAN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LPAN, self).__init__()
        numf=64
        self.conv0 = nn.Conv2d(in_ch, numf, (3, 3), (1, 1), padding=1)

        self.FeatureExtraction1 = FeatureExtraction(out_ch, numf)
        # self.ImageReconstruction1 = ImageReconstruction(in_ch, out_ch)
        self.conv1 = nn.Conv2d(numf, out_ch, (3, 3), (1, 1), padding=1)


    def forward(self, LR):
        
        LR1 = self.conv0(LR)
        
        convt_F1 = self.FeatureExtraction1(LR1)
        # HR_2 = self.ImageReconstruction1(LR, convt_F1)
        HR_2 = self.conv1(convt_F1)
        
        return HR_2

class LSTMUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(LSTMUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=False)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, (cur_hidden, cur_cell) = self.lstm(output)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden, cur_cell    

class LPAN_LSTM(nn.Module):
    def __init__(self, in_ch, out_ch, features):
        super(LPAN_LSTM, self).__init__()
        numf=64
        numf0 = 64
        # self.conv0 = nn.Conv2d(in_ch, numf, (3, 3), (1, 1), padding=1)
        # self.conv00 = nn.Conv2d(numf, numf0, (3, 3), (1, 1), padding=1)

        self.conv1 = nn.Conv2d(numf0, out_ch, (3, 3), (1, 1), padding=1)

        self.FeatureExtraction1 = FeatureExtraction(out_ch, numf)
        embed_dim = features
        hidden_size = 64
        self.num_layers = 1
        # self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm = LSTMUnit(features, embed_dim, hidden_size, num_layers = self.num_layers)
        
        self.conv0 = conv_block1(in_ch, numf,1,1,1)
        
        self.bn1 = nn.BatchNorm2d(in_ch)

    def forward(self, LR):
        
        # LR = self.bn1(LR)
        BATCH_SIZE, seq_len, H, W = LR.shape
        
        convt_F1 = LR.flatten(2)
        convt_F1, hn, cn = self.lstm(convt_F1)
        convt_F1 = rearrange(convt_F1, 'b c (h w) -> b c h w', h= H)
        
        convt_F1 = self.bn1(convt_F1)+LR
        
        LR1 = self.conv0(convt_F1)
        
        convt_F1 = self.FeatureExtraction1(LR1)
        # convt_F1 = self.conv00(convt_F1)
        
        # convt_F1 = convt_F1.flatten(2)
        # convt_F1, hn, cn = self.lstm(convt_F1)
        # convt_F1 = rearrange(convt_F1, 'b c (h w) -> b c h w', h= H)
        HR_2 = self.conv1(convt_F1)
        
        return HR_2
        

# if __name__ == '__main__':
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input=torch.randn(50,7,512).to(device)
#     a2 = TemporalConvNet(512,256, 5,[64, 128, 256]).to(device)
#     output=a2(input)
#     print(output.shape)        
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:14:45 2021

@author: 5106
"""


import math

import pylab
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import h5py
import torch.utils.data as Data
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from einops import rearrange

# from metrics import NMSELoss, Adap_NMSELoss
from models.model import CNN_LSTM, Informer, InformerStack, InformerL,InformerSL,InformerS,LSTM,LSTM1,RNN,GRU, InformerStack_e2e
from models.TSModels import Linear, DLinear, NLinear, TemporalConvNet, A2Attention, LPAN, LPAN_LSTM, TemporalConvNet2D
from models.TSAttention import MobileViT

import matplotlib.pyplot as plt
from thop import profile
from thop import clever_format
from einops import rearrange

from model_UMLP import *
from FreTS import Model_Fre


# from pvec import pronyvec
# from PAD import PAD3

import argparse

M = 32
N = 256

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = '0')
parser.add_argument('--use_gpu', type = bool, default = 0)
parser.add_argument('--gpu_list',  type = str,  default='0', help='input gpu list')

parser.add_argument('--SNR', type = float, default = 10)


parser.add_argument('--seq_len', type=int, default = 10, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default = 5, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default = 5, help='prediction sequence length')

parser.add_argument('--batch',  type= int,  default = 64)
parser.add_argument('--samples',  type= int,  default = 1)
parser.add_argument('--ir_test',  type= int,  default = 1)
parser.add_argument('--v_max',  type= int,  default = 60)
parser.add_argument('--v_min',  type= int,  default = 30)


# LSTM 
parser.add_argument('--hs', type = int, default = 256)
parser.add_argument('--hl', type = int, default = 2)
# informer
parser.add_argument('--enc_in', type=int, default=M*N*2, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=M*N*2, help='decoder input size')
parser.add_argument('--c_out', type=int, default=M*N*2, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
# parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--embed', type=str, default='fixed', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')


args = parser.parse_args()


# Parameters Setting for Training
gpu_list = args.gpu_list

# 使用GPU
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# informerL = InformerL(
#     args.enc_in,
#     args.dec_in, 
#     args.c_out, 
#     args.seq_len, 
#     args.label_len,
#     args.pred_len, 
#     args.factor,
#     64, #args.d_model, 
#     args.n_heads, 
#     args.e_layers,
#     args.d_layers, 
#     args.d_ff,
#     args.dropout, 
#     args.attn,
#     args.embed,
#     args.activation,
#     args.output_attention,
#     args.distil,
#     device
# )



# # model structure
# informer_e2e = InformerStack_e2e(
#     args.enc_in,
#     args.dec_in, 
#     args.c_out, 
#     args.seq_len, 
#     args.label_len,
#     args.pred_len, 
#     args.factor,
#     64, 
#     args.n_heads, 
#     args.e_layers,
#     args.d_layers, 
#     args.d_ff,
#     args.dropout, 
#     args.attn,
#     args.embed,
#     args.activation,
#     args.output_attention,
#     args.distil,
#     device
# )



# gru = GRU(args.enc_in, args.enc_in, args.hs, args.hl).to(device)
# linear = Linear(args.seq_len, args.pred_len,args.enc_in, 1).to(device)
# Dlinear = DLinear(args.seq_len, args.pred_len,args.enc_in, 1).to(device)
# Nlinear = NLinear(args.seq_len, args.pred_len,args.enc_in, 1).to(device)

np.random.seed(0)

# 计算NMSE
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
    x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_C = x_real  + 1j * (x_imag )
    x_hat_C = x_hat_real  + 1j * (x_hat_imag )
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse
    
index_Net = 5   # 网络选择指示器
index_LSTM = 1
deep_supervision = 0
regularization = 0

flag = 0 # 输入数据为3维张量
if index_Net ==1:
    model = MobileViT(args.seq_len, args.label_len).to(device)
    # regularization = 1

elif index_Net ==2:
    flag =1 # 输入数据为3维张量
    lstm = LSTM(args.enc_in, args.enc_in, args.hs, args.hl).to(device)
    cnn_lstm = CNN_LSTM(args.seq_len, args.enc_in, args.enc_in, args.hs, args.hl).to(device)
    rnn =  RNN(args.enc_in, args.enc_in, args.hs, args.hl).to(device)

    if index_LSTM == 1:
        model = lstm.to(device)
    elif index_LSTM == 2:
        model = cnn_lstm.to(device)
        flag = 0
    else:
        model = rnn.to(device)
elif index_Net ==3:
    if index_LSTM == 1:
        model = LPAN(args.seq_len, args.label_len).to(device)
    elif index_LSTM == 2: 
        model = LPAN_LSTM(args.seq_len, args.label_len, args.enc_in).to(device)
elif index_Net ==4:
    if index_LSTM == 1:
        model = TemporalConvNet(args.enc_in,args.enc_in, args.pred_len,[64, 128, 256]).to(device)
        flag =1 # 输入数据为3维张量
    else:
        model = TemporalConvNet2D(args.seq_len, args.label_len, args.enc_in,args.enc_in, args.pred_len,[64, 128, 256]).to(device)
elif index_Net == 5:
    from UMLP_plus import *
    deep_supervision = 1
    model = channel_est_pruning(args.seq_len, args.label_len, N, M*2, deep_supervision = deep_supervision).to(device)
    
    # from UMLP_wave import *
    # model = channel_est(args.seq_len, args.label_len, N,M*2).to(device)
else:
    model = channel_est(args.seq_len, args.label_len, N,M*2).to(device)

# if flag ==0:
#     inputs = torch.randn(1,args.seq_len,N,M*2).to(device)
#     flops, params = profile(model, inputs=(inputs,))
# else:
#     inputs = torch.randn(1,args.seq_len,N*M*2).to(device)
#     flops, params = profile(model, inputs=(inputs, args.pred_len, device))
# flops, params = clever_format([flops, params], "%.3f")
# print('flops: ', flops, 'params: ', params)


# 读取数据集
# 训练集
class MyDataset(Dataset):
    def __init__(self):
        
        dataName = 'Sat_RIS_HF_R3_{}input_{}output_{}antenna_{}element.mat'.format(args.seq_len, args.pred_len,M,N) 
        path=dataName
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['output_da1']))

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['input_da1']))
            # train_y1 = train_y1.transpose([0,3,1,2])
        
        self.X = train_y1.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)

        del file
        self.len = len(self.X)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        y1 = self.Y1[idx]
        # y2 = self.Y2[idx]
        return (x, y1)
    
class MyDatasetVal(Dataset):
    def __init__(self):
        
        dataName = 'Sat_RIS_HF_R3_{}input_{}output_{}antenna_{}element.mat'.format(args.seq_len, args.pred_len,M,N) 
        path=dataName
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['output_da_test1']))

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['input_da_test1']))
            # train_y1 = train_y1.transpose([0,3,1,2])
        
        self.X = train_y1.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)

        del file
        self.len = len(self.X)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        y1 = self.Y1[idx]
        # y2 = self.Y2[idx]
        return (x, y1)

# 读取测试集
class MyDataset1(Dataset):
    def __init__(self):
        
        dataName = 'Sat_RIS_HF_R3_test_{}input_{}output_{}antenna_{}element.mat'.format(args.seq_len, args.pred_len,M,N) 
        path=dataName
        # path="Sat_test_25input_5output_256antenna.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['Hd1']))

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['Yd1']))

        self.X = train_y1.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)

        del file
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        y1 = self.Y1[idx]
        return (x, y1)

BATCH_SIZE=16
train_dataset = MyDataset()  
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)  # shuffle

test_BATCH_SIZE=20
test_dataset = MyDatasetVal()
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_BATCH_SIZE,
                            shuffle=False,drop_last=True)  # shuffle 

del train_dataset
del test_dataset      

loss_func = nn.L1Loss().to(device)
loss_nmse = NMSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07,weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4,nesterov=True)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=1e-3, momentum=0.9, centered=False)

epochs = 50   
cost1tr = []
cost2tr = []
cost1D = []
cost2D = []
cost1ts = []
cost2ts = []
costtr = []
costD = []
costts = []

tr_nmse2 = []
tr_nmse3 = []
tr_nmse4 = []
nm1=[]
nm2=[]

# 余弦学习率下降规则
def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# 定义L1正则化函数
def l1_regularizer(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

# 计算正则化的损失

# 网络训练
if index_Net ==1:
    modelName = 'Informer_R3.pth'
elif index_Net ==2:
    modelName = 'LSTM_R3.pth'
elif index_Net ==3:
    modelName = 'CNN_LSTM_R3.pth'
elif index_Net ==4:
    modelName = 'TCN_R3.pth'
elif index_Net ==5:
    modelName = 'Uplus_R3.pth'
else:
    modelName = 'uMLP_R3.pth'
# model = torch.load(modelName).to(device)


index_flop=0 

data_aug = 0

bestLoss = 10

# model = torch.load(modelName).to(device)

for it in range(epochs):
    lr = adjust_learning_rate(optimizer, it,1e-3,1e-5)
    model.train()
    epoch_cost = 0
    epoch_cost1 = 0
    epoch_cost2 = 0
    mb_size = 32
    iteration =0
    for i, (x, y1) in enumerate(train_loader):
        a,b,c,d=x.size()
        XE, YE1= x.to(device), y1.to(device)
        
        if data_aug == 1:
            mix_input, mix_label = cutmixup(
                XE, YE1,
                mixup_prob=0.1, mixup_alpha=1.2,
                cutmix_prob=0.1, cutmix_alpha=0.7
            )

            # mix_input, mix_label = mixup(mix_input, mix_label, prob=0.1, alpha=0.4)

            # mix_input, mix_label = cutmix(mix_input, mix_label, prob=0.4, alpha=0.7)

            # modelInput, label = rgb(mix_input, mix_label, prob=0.2)
            XE, YE1 = rgb1(mix_input, mix_label, prob=0.1)
        
        if flag == 1:
            XE = rearrange(XE, 'b c h w -> b c (h w)')
            YE1 = rearrange(YE1, 'b c h w -> b c (h w)')

        
        # 如果使用 Transfomer, 网络输入数据
        if index_Net ==1:
            Yhat1 = model(XE)
            
            # 统计计算复杂度
            if index_flop==1:
                flops, params = profile(model, inputs=(XE[0,:].unsqueeze(0),))
        
        # 如果使用 LSTM, 网络输入数据
        elif index_Net ==2:
            Yhat1 = model.test_data(XE, args.pred_len, device)
            if index_flop==1:
                flops, params = profile(model, inputs=(XE[0,:].unsqueeze(0), args.pred_len, device))
        
        # 如果使用线性网络，网络输入数据
        else:
            # mix_input, mix_label = cutmixup(
            #     XE, YE1,
            #     mixup_prob=0.2, mixup_alpha=1.2,
            #     cutmix_prob=0.2, cutmix_alpha=0.7
            # )

            # mix_input, mix_label = mixup(mix_input, mix_label, prob=0.2, alpha=0.4)

            # # mix_input, mix_label = cutmix(mix_input, mix_label, prob=0.4, alpha=0.7)

            # # modelInput, label = rgb(mix_input, mix_label, prob=0.2)
            # XE, YE1 = rgb1(mix_input, mix_label, prob=0.2)
            
            Yhat1 = model(XE)
            
        if deep_supervision:
            # Yhat1 = model(XE)
            loss = 0
            for output in Yhat1:
                # loss += loss_func(output, YE1)
                
                # 原时域损失
                # loss_tmp = ((output-YE1)**2).mean()
                loss_tmp = loss_func(output, YE1)

                # 所提频域损失
                loss_feq = (torch.fft.rfft(output, dim=1) - torch.fft.rfft(YE1, dim=1)).abs().mean() 
                # 注释1. 频域损失可与时域损失加权融合，也可单独使用，一般均有性能提升，见灵敏度实验部分。
                # 注释2. 频域损失使用MAE而不是MSE，是因为不同频谱分量的量级相差非常大。使用MSE会进一步放大这种差异，导致优化过程不稳定。 
                loss += 0.6 * loss_tmp + 0.4 * loss_feq

                
            loss /= len(Yhat1)
            
            Yhat1 = Yhat1[-1]
        # else:
        #     Yhat1 = model(XE)
        
        
    #     if index_flop==1:
    #         flops, params = profile(model, inputs=(XE[0,:].unsqueeze(0),))
    # if index_flop==1:
    #     flops, params = clever_format([flops, params], "%.3f")
    #     print('flops: ', flops, 'params: ', params)
    
    # if deep_supervision:
    #     Yhat1 = Yhat1[-1]
        else:
            loss = loss_func(Yhat1, YE1)
            
        optimizer.zero_grad()
        
        if regularization ==1:
            regularization_loss = l1_regularizer(model, lambda_l1=0.00001)
            total_loss = loss + regularization_loss
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        else:
            loss.backward()
            
        optimizer.step()
        
        if i%50==0:
            nmsei=np.zeros([a, 1])
            for i1 in range(a):
                nmsei[i1] = np.sum(np.square(np.abs(Yhat1[i1,:,:].cpu().detach().numpy()-YE1[i1,:,:].cpu().numpy()))) / np.sum(np.square(np.abs(YE1[i1,:,:].cpu().numpy())))
                tr_nmse = np.sum(nmsei) / a 
                
            print("===> Epoch[{}]({}/{}) nmse: {:.4f}".format(
            it, i, len(train_loader),10*np.log10(tr_nmse)))
        
        # epoch_cost = epoch_cost + (loss / BATCH_SIZE)
        epoch_cost = epoch_cost + loss

        
    costtr.append(epoch_cost/(i+1))
    print('Iter-{}; Total loss: {:.4}'.format(it, epoch_cost.item()))

    with torch.no_grad():
        model.eval()
        epoch_cost = 0
        tr_nmse1 = []
        for i, (x, y1) in enumerate(test_loader):
            
            XE, YE1 = x.to(device), y1.to(device)
            
            if flag == 1:
                XE = rearrange(XE, 'b c h w -> b c (h w)')
                YE1 = rearrange(YE1, 'b c h w -> b c (h w)')
            
            # 如果使用 Transfomer, 网络输入数据
            if index_Net ==1:
                # inp_net = XE
                # enc_inp = inp_net
                # dec_inp =  torch.zeros_like( inp_net[:, -args.pred_len:, :] ).to(device)
                # dec_inp =  torch.cat([inp_net[:, args.seq_len - args.label_len:args.seq_len, :], dec_inp], dim=1)
                # Yhat1 = model(enc_inp, dec_inp)
                
                Yhat1 = model(XE)
            
            # 如果使用 LSTM, 网络输入数据
            elif index_Net ==2:
                Yhat1 = model.test_data(XE, args.pred_len, device)
            
            # 如果使用线性网络，网络输入数据
            else:
                Yhat1 = model(XE)
                
            if deep_supervision:
                Yhat1 = Yhat1[-1]
                
            loss = loss_func(Yhat1, YE1)

            # epoch_cost = epoch_cost + (loss / test_BATCH_SIZE)
            epoch_cost = epoch_cost + loss


            # 计算NMSE
            nmsei1=np.zeros([YE1.shape[0], 1])
            for i1 in range(YE1.shape[0]):
                nmsei1[i1] = np.sum(np.square(np.abs(Yhat1[i1,:].cpu().detach().numpy()-YE1[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,:].cpu().detach().numpy())))
                
            tr_nmse1.append(np.mean(nmsei1))
            
        nm1.append(np.mean(tr_nmse1))
        costD.append(torch.mean(epoch_cost)/(i+1)) 

        print('Iter-{}; valLoss: {:.4}; NMSE: {:.4}'.format(it, torch.mean(epoch_cost)/(i+1), 10*np.log10(np.mean(tr_nmse1))))
        
        # 保存最优模型
        if np.mean(tr_nmse1) < bestLoss:
            # if index_Net ==1:
            #     modelName = 'Informer_epoch{}.pth'.format(it)
            # elif index_Net ==2:
            #     modelName = 'LSTM_epoch{}.pth'.format(it)
            # else:
            #     modelName = 'Linear_epoch{}.pth'.format(it)
                
            
                
            torch.save(model, modelName)
            print("Model saved")
            bestLoss = np.mean(tr_nmse1) 
            
del train_loader


test1_nmse=[]
test2_nmse=[]
test3_nmse=[]
test4_nmse=[]
nmse1_snr=[]
nmse2_snr=[]
nmse3_snr=[]
nmse4_snr=[]

model = torch.load(modelName).to(device)

test_BATCH_SIZE=50
test_dataset = MyDataset1()
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_BATCH_SIZE,
                            shuffle=False, drop_last=True)  # shuffle 标识要打乱顺序
del test_dataset

with torch.no_grad():
    model.eval()
    for i, (x, y1) in enumerate(test_loader):
        XE, YE1= x.to(device), y1.to(device)
        
        if flag == 1:
            XE = rearrange(XE, 'b c h w -> b c (h w)')
            YE1 = rearrange(YE1, 'b c h w -> b c (h w)')
        
        # 如果使用 Transfomer, 网络输入数据
        if index_Net ==1:
            # inp_net = XE
            # enc_inp = inp_net
            # dec_inp =  torch.zeros_like( inp_net[:, -args.pred_len:, :] ).to(device)
            # dec_inp =  torch.cat([inp_net[:, args.seq_len - args.label_len:args.seq_len, :], dec_inp], dim=1)
            # Yhat1 = model(enc_inp, dec_inp)
            
            Yhat1 = model(XE)
        
        # 如果使用 LSTM, 网络输入数据
        elif index_Net ==2:
            Yhat1 = model.test_data(XE, args.pred_len, device)
        
        # 如果使用线性网络，网络输入数据
        else:
            Yhat1 = model(XE)
            
        if deep_supervision:
            Yhat1 = Yhat1[-1]
        
        nmsei1=np.zeros([YE1.shape[0], YE1.shape[1]])
        for i1 in range(YE1.shape[0]):
            for i2 in range(YE1.shape[1]):
                nmsei1[i1,i2] = np.sum(np.square(np.abs(Yhat1[i1,i2,:].cpu().detach().numpy()-YE1[i1,i2,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,i2,:].cpu().detach().numpy())))
        nmse1 =np.mean(nmsei1,axis=0)
        
        test1_nmse.append(nmse1)
        if (i+1)%6==0:
            nmse1_snr.append(np.mean(test1_nmse,axis=0))
            test1_nmse=[]

# 绘制NMSE结果图
# NMSE v.s. SNR
a=np.mean(nmse1_snr,axis=1)
nmse1_db=10*np.log10(np.mean(nmse1_snr,axis=1))
# nmse1_db=10*np.log10(nmse1_snr[(0,4)])
snrs = np.linspace(-20,20,9)
plt.plot(snrs, nmse1_db,ls='-', marker='+', c='black',label='Linear')
plt.legend()
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.show()

# NMSE v.s. slots
plt.figure
nmse1_db_s=10*np.log10(nmse1_snr[4])
slots = np.linspace(1,args.pred_len,args.pred_len)
plt.plot(slots, nmse1_db_s,ls='-', marker='+', c='black',label='LSTM')
plt.legend()
plt.grid(True) 
plt.xlabel('slots')
plt.ylabel('NMSE/dB')
plt.show()

# Loss v.s. epochs
plt.figure
nmse1_db_s=torch.tensor(costtr).cpu()
slots = np.linspace(1,epochs,epochs)
plt.plot(slots, nmse1_db_s,ls='-', marker='+', c='black',label='LSTM')
plt.legend()
plt.grid(True) 
plt.xlabel('Training epochs')
plt.ylabel('Loss')
plt.show()

# import scipy.io as sio # mat


# if index_Net ==1:
#     dataName_trloss = 'Informer1{}_in_{}_out_{}_fea_{}_R3_trloss.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)
#     dataName_valloss = 'Informer1{}_in_{}_out_{}_fea_{}_R3_valloss.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)
    
#     dataName_slot = 'Informer1{}_in_{}_out_{}_fea_{}_R3_slot.mat'.format(data_aug, args.seq_len, args.pred_len, args.enc_in//2)

# elif index_Net ==2:
#     dataName_trloss = 'LSTM1{}_in_{}_out_{}_fea_{}_R3_trloss.mat'.format(data_aug, args.seq_len, args.pred_len, args.enc_in//2)
#     dataName_valloss = 'LSTM1{}_in_{}_out_{}_fea_{}_R3_valloss.mat'.format(data_aug, args.seq_len, args.pred_len, args.enc_in//2)

#     dataName_slot = 'LSTM1{}_in_{}_out_{}_fea_{}_R3_slot.mat'.format(data_aug, args.seq_len, args.pred_len, args.enc_in//2)
    
# elif index_Net ==3:
#     dataName_trloss = 'CNN-LSTM1{}_in_{}_out_{}_fea_{}_R3_trloss.mat'.format(data_aug, args.seq_len, args.pred_len, args.enc_in//2)
#     dataName_valloss = 'CNN-LSTM1{}_in_{}_out_{}_fea_{}_R3_valloss.mat'.format(data_aug, args.seq_len, args.pred_len, args.enc_in//2)

#     dataName_slot = 'CNN-LSTM1{}_in_{}_out_{}_fea_{}_R3_slot.mat'.format(data_aug,args.seq_len, args.pred_len, args.enc_in//2)
    
# elif index_Net ==4:
#     dataName_trloss = 'TCN1{}_in_{}_out_{}_fea_{}_R3_trloss.mat'.format(data_aug,args.seq_len, args.pred_len, args.enc_in//2)
#     dataName_valloss = 'TCN1{}_in_{}_out_{}_fea_{}_R3_valloss.mat'.format(data_aug,args.seq_len, args.pred_len, args.enc_in//2)
    
#     dataName_slot = 'TCN1{}_in_{}_out_{}_fea_{}_R3_slot.mat'.format(data_aug,args.seq_len, args.pred_len, args.enc_in//2)

# elif index_Net ==5:
#     dataName_trloss = 'Uplus1{}_in_{}_out_{}_fea_{}_R3_trloss.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)
    
#     dataName_valloss = 'Uplus1{}_in_{}_out_{}_fea_{}_R3_valloss.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)
#     dataName_slot = 'Uplus1{}_in_{}_out_{}_fea_{}_R3_slot.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)

# else:
#     dataName_trloss = 'uMLP1{}_in_{}_out_{}_fea_{}_R3_trloss.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)
#     dataName_valloss = 'uMLP1{}_in_{}_out_{}_fea_{}_R1_valloss.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)

#     dataName_slot = 'uMLP1{}_in_{}_out_{}_fea_{}_R1_slot.mat'.format(data_aug,args.seq_len, args.pred_len,args.enc_in//2)

    
# sio.savemat(dataName_trloss, {'a':torch.tensor(costtr).cpu().numpy()})
# sio.savemat(dataName_valloss, {'a':torch.tensor(costD).cpu().numpy()})

# sio.savemat(dataName_slot, {'a':nmse1_snr})

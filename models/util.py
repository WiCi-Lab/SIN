import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os 
import time 
import torch
from scipy.fftpack import fft, ifft 
from scipy.interpolate import interp1d
pi = np.pi

# def get_rate(H, sigma2):
#     a1 = H[0,0]
#     a2 = H[1,1]
#     I1 = H[0,1]
#     I2 = H[1,0]
#     rate1 = np.log2( 1 + np.abs(a1)**2 / ( np.abs(I1)**2 + sigma2) )
#     rate2 = np.log2( 1 + np.abs(a2)**2 / ( np.abs(I2)**2 + sigma2) )
#     rate = rate1 + rate2
#     return rate 
def get_rate(H, sigma2):
    rate = np.log2(np.linalg.det(np.eye(2) + 1/sigma2 * H.T.conj().dot(H)))
    return rate

def get_zf_rate(H_hat, H_true, SNR):
    D_np = get_zf_precoder(H_hat)
    HF = np.matmul(D_np, H_true)
    rate = SNR_rate(HF, SNR)
    return rate

def get_zf_precoder(H_hat):
    D_np = np.linalg.pinv(H_hat) # shape = [64, 2, 4]
    D_np = D_np / np.linalg.norm(D_np, axis=(2), keepdims=True)
    return D_np

def SNR_rate(H, SNR):
    rate = np.mean(np.log2(np.linalg.det(np.eye(2) + (10**(SNR/10)) * np.matmul(H.conj().transpose(0,2,1), H))))
    return rate

def SINR_rate(HF, SNR):
    HF = torch.pow(torch.abs(torch.from_numpy(HF).cuda()), 2)
    HF_diag = HF * torch.eye(2).cuda()                
    rate = torch.mean(torch.sum(torch.log2(1 + torch.sum(HF_diag, 2)\
        /(torch.abs(torch.sum(HF - HF_diag, 2))+ 1/(10**(SNR/10)))),1))
    return rate

def interpolate(H_prev, H_pred, ir):
    M, pred_len, N = H_pred.shape 
    _, prev_len, N = H_prev.shape 
    H = np.concatenate([H_prev, H_pred],  1)
    x = np.arange((pred_len + prev_len - 1) * 5 + 1)
    x0 = np.arange(pred_len + prev_len) *  ir
    x1_1 = np.arange(prev_len) *  ir
    x1_2 = np.arange( (prev_len - 1) *  ir + 1, (prev_len + pred_len -1) *  ir + 1)
    x1 = np.concatenate([x1_1, x1_2])

    H_interp = np.zeros([M, x1.size, N], dtype = np.complex)
    for i in range(M):
        for j in range(N):
            f = interp1d(x0, H[i, :, j], kind = 'cubic')
            H_interp[i, :, j] = f(x1)

    # plt.figure()
    # plt.plot(x, data[0,:,0,0].real, '--')
    # plt.plot(x0, H[0,:,0].real, '+')
    # plt.plot(x1, H_interp[0,:,0].real)
    # plt.plot(x1_2, H_interp[0,- pred_len *  ir:,0].real)
    # plt.savefig('test.png')
    return H_interp[:, - pred_len *  ir:, :]

def complex2real(data):
    B, P, N = data.shape
    data2 = data.reshape([B, P, N, 2])
    data2[...,0] = data.real 
    data2[...,1] = data.imag    
    return data2

def real2complex(data):
    B, P, N = data.shape 
    data2 = data.reshape([B, P, N//2, 2])
    data2 = data2[:,:,:,0] + 1j * data2[:,:,:,1]
    return data2



def get_result(tensor, Nt = 4, Nr = 2):
    # tensor shape: Batch * seq_len * 2 * subcarrier * (Nt \times Nr)
    result = np.array(tensor)
    result = result[:,:,0,:,:] + 1j * result[:,:,0,:,:]
    shape = list(result.shape)[:-1] 
    shape.extend([Nr, Nt])
    # print(shape)
    result = result.reshape(shape)
    return result 


def Torch_Complex_Matrix_Matmul(A, B):
    Ar = A[:,:,:,:,0]
    Ai = A[:,:,:,:,1]
    Br = B[:,:,:,:,0]
    Bi = B[:,:,:,:,1]
    A1 = torch.cat([torch.cat([Ar,-Ai],3),torch.cat([Ai,Ar],3)],2)
    B1 = torch.cat([Br,Bi],2)
    C = torch.matmul(A1,B1)    
    C = torch.cat((C[:,:,:int(C.size(2)/2),:].unsqueeze(4), C[:,:,int(C.size(2)/2):,:].unsqueeze(4)),4)
    return C


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


def get2DDFT(Nx, Ny):
    az =  np.linspace(-1/2 + 1/Nx, 1/2, Nx).reshape(1,Nx)
    el =  np.linspace(-1/2 + 1/Ny, 1/2, Ny).reshape(1,Ny)
    A_az = np.exp(-1j * 2 * pi * (np.arange(Nx).reshape(Nx,1)).dot(az))
    A_el = np.exp(-1j * 2 * pi * (np.arange(Ny).reshape(Ny,1)).dot(el))
    A = np.kron(A_az, A_el)/np.sqrt(Nx*Ny)
    return A






        
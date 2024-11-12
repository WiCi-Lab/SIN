'''
The content of this file could be self-defined. 
But please note the interface of the following function cannot be modified,
    - encFunction_1
    - decFunction_1
    - encFunction_2
    - decFunction_2
'''
#=======================================================================================================================
#=======================================================================================================================
# Package Importing
import random
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from einops.layers.torch import Rearrange
from einops import rearrange
from collections import OrderedDict
#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Function Defining
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)
    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2
    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)
def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None
class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out
class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out
#=======================================================================================================================
#=======================================================================================================================
# Eigenvector Calculation Function Defining
def cal_eigenvector(channel):
    """
        Description:
            calculate the eigenvector on each subband
        Input:
            channel: np.array, channel in frequency domain,  shape [batch_size, rx_num, tx_num, subcarrier_num]
        Output:
            eigenvectors:  np.array, eigenvector for each subband, shape [batch_size, tx_num, subband_num]
    """
    subband_num = 13
    hf_ = np.transpose(channel, [0,3,1,2]) # (batch,subcarrier_num,4,32)
    hf_h = np.conj(np.transpose(channel, [0,3,2,1])) # (batch,subcarrier_num,32,4)
    R = np.matmul(hf_h, hf_) # (batch,subcarrier_num,32,32)
    R = R.reshape(R.shape[0],subband_num,-1,R.shape[2],R.shape[3]).mean(axis=2) # average the R over each subband, (batch,13,32,32)
    [D,V] = np.linalg.eig(R)
    v = V[:,:,:,0]
    eigenvectors = np.transpose(v,[0,2,1])
    return eigenvectors
#=======================================================================================================================
#=======================================================================================================================
# Loss Function Defining
class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
    def forward(self, w_true, w_pre):
        cos_similarity = cosine_similarity_cuda(w_true.permute(0, 3, 2, 1), w_pre.permute(0, 3, 2, 1))
        if self.reduction == 'mean':
            cos_similarity_scalar = cos_similarity.mean()
        elif self.reduction == 'sum':
            cos_similarity_scalar = cos_similarity.sum()
        return 1 - cos_similarity_scalar
def cosine_similarity_cuda(w_true, w_pre):
    num_batch, num_sc, num_ant = w_true.size(0), w_true.size(1), w_true.size(2)
    w_true = w_true.reshape(num_batch * num_sc, num_ant, 2)
    w_pre = w_pre.reshape(num_batch * num_sc, num_ant, 2)
    w_true_re, w_true_im = w_true[..., 0], w_true[..., 1]
    w_pre_re, w_pre_im = w_pre[..., 0], w_pre[..., 1]
    numerator_re = (w_true_re * w_pre_re + w_true_im * w_pre_im).sum(-1)
    numerator_im = (w_true_im * w_pre_re - w_true_re * w_pre_im).sum(-1)
    denominator_0 = (w_true_re ** 2 + w_true_im ** 2).sum(-1)
    denominator_1 = (w_pre_re ** 2 + w_pre_im ** 2).sum(-1)
    cos_similarity = torch.sqrt(numerator_re ** 2 + numerator_im ** 2) / (
                torch.sqrt(denominator_0) * torch.sqrt(denominator_1))
    cos_similarity = cos_similarity ** 2
    return cos_similarity
#=======================================================================================================================
#=======================================================================================================================
# Data Loader Class Defining
class DatasetFolder(Dataset):
    def __init__(self, matInput, matLabel):
        self.input, self.label = matInput, matLabel
    def __getitem__(self, index):
        return self.input[index], self.label[index]
    def __len__(self):
        return self.input.shape[0]
class DatasetFolder_mixup(Dataset):
    def __init__(self, matInput, matLabel):
        self.input, self.label = matInput, matLabel
    def __getitem__(self, index):
        mixup_ratio = 0.3
        r = np.random.rand(1)
        if r < mixup_ratio:
            mix_idx = random.randint(0, self.input.shape[0]-1)
            lam = np.random.rand(1)
            mix_input = np.zeros(self.input[index].shape, dtype='float32')
            mix_label = np.zeros(self.label[index].shape, dtype='float32')
            mix_input[:] = lam * self.input[index] + (1-lam) * self.input[mix_idx]
            mix_label[:] = lam * self.label[index] + (1-lam) * self.label[mix_idx]
        else:
            mix_input, mix_label = self.input[index], self.label[index]
        return mix_input, mix_label
    def __len__(self):
        return self.input.shape[0]
class DatasetFolder_eval(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]
#==================================
#=======================================================================================================================
#=======================================================================================================================
# Model Defining
# Channel Estimation Class Defining
class channel_est(nn.Module):
    def __init__(self, input_length):
        super(channel_est, self).__init__()
        pilot_num = int(input_length / 8)
        emd_dim = 192
        out_dim = 8
        patch_size_h=1
        patch_size_w=2
        image_size_h=32
        image_size_w=52
        num_patch =  (image_size_h// patch_size_h) * (image_size_w// patch_size_w)
        self.mlp_mixer = MLPMixer(in_channels=8, dim=emd_dim, num_output=out_dim, 
                                patch_size_h=patch_size_h, patch_size_w=patch_size_w, image_size_h=image_size_h, image_size_w=image_size_w, depth=8, token_dim=512, channel_dim=512)
        self.fc1 = nn.Linear(pilot_num, image_size_w)
        self.fc2 = nn.Linear(num_patch, image_size_h*image_size_w)
    def forward(self, x): #(batch, 2, 4, 208 or 48, 4)
        x = rearrange(x, 'b c rx (rb subc) sym -> b (c rx) (subc sym) rb', subc=8) # (batch,8,32,26 or 6)
        x = self.fc1(x) # (batch,8,32,52)
        out = self.mlp_mixer(x) # (batch, patch_num, out_dim)
        out = rearrange(out, 'b p o -> b o p')
        out = self.fc2(out) # (batch, 8, 52*32)
        out = rearrange(out, 'b (c rx) (tx rb) -> b c rx tx rb', rx=4, tx=32)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x
class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_output, patch_size_h, patch_size_w, image_size_h, image_size_w, depth, token_dim, channel_dim):
        super().__init__()
        assert image_size_w % patch_size_w == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size_h % patch_size_h == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size_h// patch_size_h) * (image_size_w// patch_size_w)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, (patch_size_h, patch_size_w), (patch_size_h, patch_size_w)),
            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(256, dim, kernel_size=3, stride=1, padding=1, bias=True),
            Rearrange('b c h w -> b (h w) c'),
            # nn.LayerNorm(dim)
        )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_output)
        )
    def forward(self, x): # (batch, 8, 26, 32)
        x = self.to_patch_embedding(x) # (batch, patch, dim)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x) # (batch, patch, dim)
        #x = x.mean(dim=1)
        return self.mlp_head(x) 
# Encoder and Decoder Class Defining
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x
class Encoder1(nn.Module):
    B = 2
    def __init__(self, feedback_bits, quantization=True):
        super(Encoder1, self).__init__()
        d_model = 384 
        nhead = 6
        d_hid = 512 
        dropout = 0.0
        nlayers = 4
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.fc1 = nn.Linear(64, d_model)
        self.fc2 = nn.Linear(13*d_model, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization
    def forward(self, x): # (batch, 2, 32, 13)
        x = rearrange(x, 'b c eig f -> f b (c eig)')
        out = self.fc1(x) # (13, batch, d_model)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        out = rearrange(out, 'f b dmodel -> b (f dmodel)')
        out = self.fc2(out) # (batch, 512/B)
        out = self.sig(out)
        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                padding=padding, groups=groups, bias=True))
        ]))
class Decoder1(nn.Module):
    B = 2
    def __init__(self, feedback_bits, quantization=True):
        super(Decoder1, self).__init__()
        d_model = 384 
        nhead = 6
        d_hid = 512 
        dropout = 0.0
        nlayers = 4
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers1 = nn.TransformerEncoderLayer(d_model, nhead, 512, dropout)
        self.transformer_decoder1 = nn.TransformerEncoder(decoder_layers1, 1)
        decoder_layers11 = nn.TransformerEncoderLayer(d_model, nhead, 384, dropout)
        self.transformer_decoder11 = nn.TransformerEncoder(decoder_layers11, 1)
        decoder_layers2 = nn.TransformerEncoderLayer(d_model, nhead, 256, dropout)
        self.transformer_decoder2 = nn.TransformerEncoder(decoder_layers2, 1)
        decoder_layers21 = nn.TransformerEncoderLayer(d_model, nhead, 512, dropout)
        self.transformer_decoder21 = nn.TransformerEncoder(decoder_layers21, 1)
        decoder_layers22 = nn.TransformerEncoderLayer(d_model, nhead, 384, dropout)
        self.transformer_decoder22 = nn.TransformerEncoder(decoder_layers22, 1)
        decoder_layers3 = nn.TransformerEncoderLayer(d_model, nhead, 512, dropout)
        self.transformer_decoder3 = nn.TransformerEncoder(decoder_layers3, 1)
        self.sig = nn.Sigmoid()
        self.quantization = quantization 
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc1 = nn.Linear(int(feedback_bits / self.B), 13*d_model)
        self.fc20 = nn.Linear(d_model, 32*2)
        self.fc21 = nn.Linear(d_model, 32*2)
        self.fc22 = nn.Linear(d_model, 32*2)
        self.fc23 = nn.Linear(d_model, 32*2)
        self.out_cov = ConvBN(8, 2, 3)
    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = self.fc1(out) # (batch, 13*d_model)
        out = rearrange(out, 'b (f dmodel) -> f b dmodel', f=13)
        out0 = self.fc20(out) # (13, batch, 32*2)
        out0 = rearrange(out0, 'f b (c eig) -> b c eig f', c=2)
        out = self.pos_encoder(out)
        out1 = self.transformer_decoder1(out) # (13, batch, d_model)
        out11 = self.transformer_decoder11(out1) # (13, batch, d_model)
        out1 = self.fc21(out11) # (13, batch, 32*2)
        out1 = rearrange(out1, 'f b (c eig) -> b c eig f', c=2)
        out2 = self.transformer_decoder2(out) # (13, batch, d_model)
        out2 = self.transformer_decoder21(out2) # (13, batch, d_model)
        out2 = self.transformer_decoder22(out2) # (13, batch, d_model)
        out2 = self.fc22(out2) # (13, batch, 32*2)
        out2 = rearrange(out2, 'f b (c eig) -> b c eig f', c=2)
        out3 = self.transformer_decoder3(out) # (13, batch, d_model)
        out3 = self.fc23(out3) # (13, batch, 32*2)
        out3 = rearrange(out3, 'f b (c eig) -> b c eig f', c=2)
        out = torch.cat((out0, out1, out2, out3), dim=1)
        out = self.out_cov(out)
        return out

class AutoEncoder1(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder1, self).__init__()
        self.encoder1 = Encoder1(feedback_bits)
        self.decoder1 = Decoder1(feedback_bits)
    def forward(self, x):
        feature = self.encoder1(x)
        out = self.decoder1(feature)
        return out
# Encoder and Decoder Class Defining
class Encoder2(nn.Module):
    B = 2
    def __init__(self, feedback_bits, quantization=True):
        super(Encoder2, self).__init__()
        d_model = 384 
        nhead = 6
        d_hid = 512 
        dropout = 0.0
        nlayers = 3
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.fc1 = nn.Linear(64, d_model)
        self.fc2 = nn.Linear(13*d_model, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization
    def forward(self, x): # (batch, 2, 32, 13)
        # x = x - 0.5
        x = rearrange(x, 'b c eig f -> f b (c eig)')
        out = self.fc1(x) # (13, batch, d_model)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        out = rearrange(out, 'f b dmodel -> b (f dmodel)')
        out = self.fc2(out) # (batch, 512/B)
        out = self.sig(out)
        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out
class Decoder2(nn.Module):
    B = 2
    def __init__(self, feedback_bits, quantization=True):
        super(Decoder2, self).__init__()
        d_model = 384 
        nhead = 6
        d_hid = 512 
        dropout = 0.0
        nlayers = 3
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, nlayers)
        self.sig = nn.Sigmoid()
        self.quantization = quantization 
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc1 = nn.Linear(int(feedback_bits / self.B), 13*d_model)
        self.fc2 = nn.Linear(d_model, 32*2)
    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = self.fc1(out) # (batch, 13*d_model)
        out = rearrange(out, 'b (f dmodel) -> f b dmodel', f=13)
        out = self.pos_encoder(out)
        out = self.transformer_decoder(out) # (13, batch, d_model)
        out = self.fc2(out) # (13, batch, 32*2)
        # out = out + 0.5
        out = rearrange(out, 'f b (c eig) -> b c eig f', c=2)
        return out
class AutoEncoder2(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder2, self).__init__()
        self.encoder2 = Encoder2(feedback_bits)
        self.decoder2 = Decoder2(feedback_bits)
    def forward(self, x):
        feature = self.encoder2(x)
        out = self.decoder2(feature)
        return out
#=======================================================================================================================
#=======================================================================================================================
# Function Defining
def encFunction_1(pilot_1, encModel_p1_1_path, encModel_p1_2_path):
    """
        Description:
            CSI compression based on received pilot signal
        Input:
            pilot_1: np.array, received pilot signal,  shape [NUM_SAMPLES, 2, rx_num, pilot on different subcarrier, pilot on different symbol]
            encModel_p1_1_path: path to load the first AI model, please ignore if not needed
            encModel_p1_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, encoded bit steam, shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
    """
    num_feedback_bits = 64
    subc_num = 208
    model_ce = channel_est(subc_num).cuda()
    model_ce.load_state_dict(torch.load(encModel_p1_1_path)['state_dict'])
    model_fb = AutoEncoder1(num_feedback_bits).cuda()
    model_fb.encoder1.load_state_dict(torch.load(encModel_p1_2_path)['state_dict'])
    test_dataset = DatasetFolder_eval(pilot_1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4, pin_memory=True)
    model_ce.eval()
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            # step 1: channel estimation
            h = model_ce(data) # (batch,2,4,32,52)
            # step 2: eigenvector calculation
            h_complex = h[:,0,...] + 1j*h[:,1,...] # (batch,4,32,52)
            h_complex = h_complex.cpu().numpy()
            v = cal_eigenvector(h_complex)
            # step 3: eigenvector compression
            w_complex = torch.from_numpy(v)
            w = torch.zeros([h.shape[0], 2, 32, 13], dtype=torch.float32).cuda() # (batch,2,32,13)
            w[:,0,:,:] = torch.real(w_complex)
            w[:,1,:,:] = torch.imag(w_complex)
            modelOutput = model_fb.encoder1(w)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all
def decFunction_1(bits_1, decModel_p1_1_path, decModel_p1_2_path):
    """
        Description:
            CSI reconstruction based on feedbacked bit stream
        Input:
            bits_1: np.array, feedbacked bit stream,  shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
            decModel_p1_1_path: path to load the first AI model, please ignore if not needed
            decModel_p1_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, reconstructed CSI (eigenvectors), shape [NUM_SAMPLES, 2, NUM_TX, NUM_SUBBAND]
    """
    num_feedback_bits = 64
    model_fb = AutoEncoder1(num_feedback_bits).cuda()
    model_fb.decoder1.load_state_dict(torch.load(decModel_p1_1_path)['state_dict'])
    test_dataset = DatasetFolder_eval(bits_1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4, pin_memory=True)
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            modelOutput = model_fb.decoder1(data)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all
def encFunction_2(pilot_2, encModel_p2_1_path, encModel_p2_2_path):
    """
        Description:
            CSI compression based on received pilot signal
        Input:
            pilot_2: np.array, received pilot signal,  shape [NUM_SAMPLES, 2, rx_num, pilot on different subcarrier, pilot on different symbol]
            encModel_p2_1_path: path to load the first AI model, please ignore if not needed
            encModel_p2_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, encoded bit steam, shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
    """
    num_feedback_bits = 64
    subc_num = 48
    model_ce = channel_est(subc_num).cuda()
    model_ce.load_state_dict(torch.load(encModel_p2_1_path)['state_dict'])
    model_fb = AutoEncoder2(num_feedback_bits).cuda()
    model_fb.encoder2.load_state_dict(torch.load(encModel_p2_2_path)['state_dict'])
    test_dataset = DatasetFolder_eval(pilot_2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4, pin_memory=True)
    model_ce.eval()
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            # step 1: channel estimation
            h = model_ce(data) # (batch,2,4,32,52)
            # step 2: eigenvector calculation
            h_complex = h[:,0,...] + 1j*h[:,1,...] # (batch,4,32,52)
            h_complex = h_complex.cpu().numpy()
            v = cal_eigenvector(h_complex)
            # step 3: eigenvector compression
            w_complex = torch.from_numpy(v)
            w = torch.zeros([h.shape[0], 2, 32, 13], dtype=torch.float32).cuda() # (batch,2,32,13)
            w[:,0,:,:] = torch.real(w_complex)
            w[:,1,:,:] = torch.imag(w_complex)
            modelOutput = model_fb.encoder2(w)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all
def decFunction_2(bits_2, decModel_p2_1_path, decModel_p2_2_path):
    """
        Description:
            CSI reconstruction based on feedbacked bit stream
        Input:
            bits_2: np.array, feedbacked bit stream,  shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
            decModel_p2_1_path: path to load the first AI model, please ignore if not needed
            decModel_p2_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, reconstructed CSI (eigenvectors), shape [NUM_SAMPLES, 2, NUM_TX, NUM_SUBBAND]
    """
    num_feedback_bits = 64
    model_fb = AutoEncoder2(num_feedback_bits).cuda()
    model_fb.decoder2.load_state_dict(torch.load(decModel_p2_1_path)['state_dict'])
    test_dataset = DatasetFolder_eval(bits_2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4, pin_memory=True)
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            modelOutput = model_fb.decoder2(data)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all
# Score Calculating
NUM_TX = 32
NUM_SAMPLES = 3000
NUM_SUBBAND = 13
def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1*num2))
    return cos.item()
def cal_score(w_true,w_pre):
    w_true = np.transpose(w_true, [0, 3, 2, 1])
    w_pre = np.transpose(w_pre, [0, 3, 2, 1])
    img_total = NUM_TX * 2
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j*W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j*W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i+1,]
        W_pre2_sample = W_pre2[i:i+1,]
        score_tmp = cos_sim(W_true2_sample,W_pre2_sample)
        score_cos = score_cos + abs(score_tmp)*abs(score_tmp)
    score_cos = score_cos/num_sample_subband
    return score_cos
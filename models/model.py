import torch
import torch.nn as nn
import torch.nn.functional as F

from util import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from einops import rearrange
from einops.layers.torch import Rearrange

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
    def __init__(self, dim, num_patch,num_output, depth, token_dim, channel_dim):
        super().__init__()
        
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_output)
        )
    def forward(self, x): # (batch, 8, 26, 32)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x) # (batch, patch, dim)
        #x = x.mean(dim=1)
        return self.mlp_head(x) 

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc,  x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack_e2e(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(InformerStack_e2e, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

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
    
# Residual module
class _Res_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.res_conb = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.instan = nn.BatchNorm2d(out_ch)

    def forward(self, x,al=1):

        y = self.relu(self.instan(self.res_conv(x)))
        y = self.res_conb(y)
        y *= al
        y = torch.add(y, x)
        return y 

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

# Axial MLP module
class MLPBlock(nn.Module):
    def __init__(self,h=224,w=224,c=3):
        super().__init__()
        self.proj_h=nn.Linear(h,h)
        self.proj_w=nn.Linear(w,w)
        self.fuse=nn.Linear(3*c,c)
        self.instan = nn.BatchNorm2d(c)
        
    
    def forward(self,x):
        x1=x
        x = self.instan(x)
        x_h=self.proj_h(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_w=self.proj_w(x)
        x_id=x
        x_fuse=torch.cat([x_h,x_w,x_id],dim=1)
        out=self.fuse(x_fuse.permute(0,2,3,1)).permute(0,3,1,2)
        return out+x1

    
# Channel Attention module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=True),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Channel-wise Convolution    
class _Res_Blocka(nn.Module):
    def __init__(self, dadis, in_ch, out_ch):
        super(_Res_Blocka, self).__init__()
        
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dadis, groups=out_ch//8,dilation=dadis)

        self.res_cona = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        
        self.ca = SELayer(out_ch)
        self.instan = nn.BatchNorm2d(out_ch)

    def forward(self, x,al=1):
        x1 = self.instan(x)
        y = self.relu(self.res_conv(x1))
        y = self.relu(self.res_cona(y))
        y = self.ca(y)
        y *= al
        y = torch.add(y, x)
        return y
    
# ConvMLP module
class sMLPBlock(nn.Module):
    def __init__(self,dadis=1,h=224,w=224,c=3):
        super().__init__()
        self.dw=_Res_Blocka(dadis,c,c)
        self.mlp=MLPBlock(h,w,c)
        self.cmlp = Mlp(in_features=c, hidden_features=c*2)
    
    def forward(self,x):
        # x1=x
        x= self.dw(x)
        out= self.mlp(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x= self.cmlp(x)+x
        out = rearrange(x, 'b h w c-> b c h w')
        return out
        
class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
        
class InformerL(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(InformerL, self).__init__()
        self.pred_len = out_len
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.projection0 = nn.Linear(enc_in, d_model, bias=True)
        self.projection2 = MLPMixer(dim= d_model, num_patch = out_len, num_output = d_model, depth=2, token_dim=64, channel_dim=64)
        self.ups = nn.Upsample(scale_factor=2,mode='linear',align_corners=True)
        
    def forward(self, x_enc):
        
        x_enc = self.ups(x_enc.permute(0,2,1)) # [B, seq_len, enc_in] ——> [B, enc_in, out_len]
        
        enc_out = self.projection0(x_enc.permute(0,2,1)) # [B, enc_in, out_len] ——> [B, out_len, d_model]
        
        enc_out = self.projection2(enc_out) # [B, out_len, d_model]
        
        dec_out = self.projection(enc_out) # [B, out_len, c_out] 
        
        return dec_out
    
class InformerSL(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(InformerSL, self).__init__()
        self.pred_len = out_len
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.projection1 = nn.Linear(seq_len, out_len, bias=True)
        self.projection0 = nn.Linear(enc_in, d_model, bias=True)
        self.projection2 = MLPMixer(dim= d_model, num_patch = seq_len, num_output = d_model, depth=3, token_dim=256, channel_dim=256)
        self.ups = nn.Upsample(scale_factor=2,mode='linear',align_corners=True)
        
    def forward(self, x_enc):
        
        enc_out = self.projection0(x_enc) # [B, enc_in, out_len] ——> [B, out_len, d_model]
        
        enc_out = self.projection2(enc_out) # [B, out_len, d_model]
        
        enc_out = self.projection1(enc_out.permute(0,2,1)) # [B, out_len, c_out] 
        dec_out = self.projection(enc_out.permute(0,2,1))
        
        
        return dec_out
    
class InformerS(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(InformerS, self).__init__()
        self.pred_len = out_len
        self.projection = nn.Conv2d(d_model, 2, kernel_size=1, stride=1, padding=0)
        # self.projection0 = nn.Linear(enc_in, d_model, bias=True)
        self.ups = nn.Upsample(size=(out_len,enc_in//2),mode='bilinear',align_corners=True)
        
        self.projection2 = sMLPBlock(dadis=1,h=out_len,w=enc_in//2,c=d_model)
        self.projection3 = sMLPBlock(dadis=1,h=out_len,w=enc_in//2,c=d_model)
        self.Conv11 = nn.Conv2d(2, d_model, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x_enc):
        
        x_enc = self.ups(x_enc) # [B, seq_len, enc_in] ——> [B, enc_in, out_len]
        x_enc = self.Conv11(x_enc)
        
        enc_out = self.projection2(x_enc) # [B, out_len, d_model]
        enc_out = enc_out + x_enc
        enc_out = self.projection3(enc_out)
        
        dec_out = self.projection(enc_out) # [B, out_len, c_out] 

        return dec_out
    

class RNNUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(RNNUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, cur_hidden = self.rnn(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden

class RNN(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.features = features
        self.model = RNNUnit(features, input_size, hidden_size, num_layers = self.num_layers)
        
    def train_data(self, x,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs
    
    def test_data(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            else:
                output, prev_hidden= self.model(output,  prev_hidden) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()

        return outputs

class GRUUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(GRUUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, cur_hidden = self.gru(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden

class GRU(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.features = features
        self.model = GRUUnit(features, input_size, hidden_size, num_layers = self.num_layers)
    

    def train_data(self, x, device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs

    def test_data(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            else:
                output, prev_hidden= self.model(output,  prev_hidden) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs

class LSTMUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(LSTMUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x, prev_hidden, prev_cell):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden, cur_cell

class LSTM(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit(features, input_size, hidden_size, num_layers = self.num_layers)
    
    def train_data(self, x,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs

    def test_data(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output,  prev_hidden, prev_cell) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()
        
        return outputs
        
    def forward(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output,  prev_hidden, prev_cell) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs
    
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
    
class CNN_LSTM(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, in_ch, features, input_size, hidden_size, num_layers = 2):
        
        super(CNN_LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Conv11 = conv_block1(in_ch, filters[0],1,2,2)
        self.Conv22 = conv_block1(filters[0], filters[1],1,1,1)
        self.Conv33 = conv_block1(filters[1], in_ch,1,1,1)
        
        self.features = features
        self.model = LSTMUnit(self.features, input_size, hidden_size, num_layers = self.num_layers)
    
    def train_data(self, x,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs

    def test_data(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, H, W = x.shape
        
        x = self.Conv11(x)
        x = self.Conv22(x)
        x = self.Conv33(x)
        
        x = rearrange(x, 'b c h w -> b c (h w)')
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output,  prev_hidden, prev_cell) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()
        
        outputs = rearrange(outputs, 'b c (h w) -> b c h w', h= H)
        
        return outputs
        
    def forward(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output,  prev_hidden, prev_cell) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs
    

class LSTMUnit1(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2,input_len=26,pred_len=6):
        
        super(LSTMUnit1, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, hidden_size))
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        self.out = nn.Linear(input_len, pred_len)
        
    def forward(self, x):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        # output1, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        output = output.permute(1,2,0)
        output = self.out(output)
        output = output.permute(0,2,1)
        
        return output

class LSTM1(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(LSTM1, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit1(features, input_size, hidden_size, num_layers = self.num_layers)
    
    # def train_data(self, x,device):
        
    #     BATCH_SIZE, seq_len, _ = x.shape
    #     prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
    #     prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
    #     outputs = []
    #     for idx in range(seq_len):
    #         output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
    #         outputs.append(output)
    #     outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        # return outputs

    def test_data(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        # prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        # prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        # outputs = []
        output = self.model(x.permute(1,0,2).contiguous())

        # outputs = output


        return output
    

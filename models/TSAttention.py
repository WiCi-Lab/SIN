# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:24:31 2024

@author: WiCi
"""

from torch import nn
import torch
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.ln=nn.BatchNorm2d(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.ln(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1),**kwargs)
    
# class PreNorm(nn.Module):
#     def __init__(self,dim,fn):
#         super().__init__()
#         self.ln=nn.LayerNorm(dim)
#         self.fn=fn
#     def forward(self,x,**kwargs):
#         return self.fn(self.ln(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,mlp_dim,dropout) :
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads,head_dim,dropout):
        super().__init__()
        inner_dim=heads*head_dim
        project_out=not(heads==1 and head_dim==dim)

        self.heads=heads
        self.scale=head_dim**-0.5

        self.attend=nn.Softmax(dim=-1)
        self.to_qkv=nn.Linear(dim,inner_dim*3,bias=False)
        
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        q,k,v=map(lambda t:rearrange(t,'b p n (h d) -> b p h n d',h=self.heads),qkv)
        dots=torch.matmul(q,k.transpose(-1,-2))*self.scale
        attn=self.attend(dots)
        out=torch.matmul(attn,v)
        out=rearrange(out,'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,head_dim,mlp_dim,dropout=0.):
        super().__init__()
        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,Attention(dim,heads,head_dim,dropout)),
                PreNorm(dim,FeedForward(dim,mlp_dim,dropout))
            ]))


    def forward(self,x):
        out=x
        for att,ffn in self.layers:
            out=out+att(out)
            out=out+ffn(out)
        return out

class MobileViTAttention(nn.Module):
    def __init__(self,in_channel=3,dim=64,kernel_size=3,patch_size1=7,patch_size2=7):
        super().__init__()
        self.ph,self.pw=patch_size1,patch_size2
        self.conv1=nn.Conv2d(in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)
        self.bn1=nn.BatchNorm2d(in_channel)
        self.conv2=nn.Conv2d(in_channel,dim,kernel_size=1)
        self.bn2=nn.BatchNorm2d(dim)

        self.trans=Transformer(dim=dim,depth=3,heads=10,head_dim=64,mlp_dim=1024)

        self.conv3=nn.Conv2d(dim,in_channel,kernel_size=1)
        self.bn3=nn.BatchNorm2d(in_channel)
        self.conv4=nn.Conv2d(2*in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)
        self.bn4=nn.BatchNorm2d(in_channel)

    def forward(self,x):
        y=x.clone() #bs,c,h,w

        ## Local Representation
        y=self.bn2(self.conv2(self.bn1(self.conv1(x)))) #bs,dim,h,w

        ## Global Representation
        _,_,h,w=y.shape
        y=rearrange(y,'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim',ph=self.ph,pw=self.pw) #bs,h,w,dim
        y=self.trans(y)
        y=rearrange(y,'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)',ph=self.ph,pw=self.pw,nh=h//self.ph,nw=w//self.pw) #bs,dim,h,w

        ## Fusion
        y=self.bn3(self.conv3(y)) #bs,dim,h,w
        y=torch.cat([x,y],1) #bs,2*dim,h,w
        y=self.bn4(self.conv4(y)) #bs,c,h,w

        return y

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

# A2Attention network
class MobileViT(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(MobileViT, self).__init__()

        n1 = 64
        filters = [n1, n1 , n1 * 4, n1 * 8, n1 * 16]
        # in_ch=8
        # out_ch=2
                
        self.Conv11 = conv_block1(in_ch, filters[0],1,2,2)
        self.Conv22 = conv_block1(filters[0], filters[1],1,1,1)
    

        self.DoubleAttention1 = MobileViTAttention(in_channel=filters[0],dim=64,kernel_size=3,patch_size1=64,patch_size2=16)
        # self.DoubleAttention2 = MobileViTAttention(in_channel=filters[1],dim=64,kernel_size=3,patch_size1=8,patch_size2=64)
        self.bn1 = nn.BatchNorm2d(filters[1])
        
        self.Conv = nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        
        e1 = self.Conv11(x) 
        e1 = self.DoubleAttention1(e1)
        e1 = self.Conv22(e1) 
        # e1 = self.DoubleAttention2(e1)
        # e1 = self.bn1(e1)
        
        out = self.Conv(e1)
            
        return out

# if __name__ == '__main__':
#     m=MobileViTAttention(in_channel=16,dim=512,kernel_size=3,patch_size1=4,patch_size2=4)
#     input=torch.randn(1,16,64,72)
#     output=m(input)
#     print(output.shape)
    
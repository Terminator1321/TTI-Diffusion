import torch
import torch.nn as nn
import torch.nn.functional as F

from DefussionBlock import ResBlock, SinusoidalTimeEmbedding, TimeEmbedding

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1)

    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1)

    def forward(self, x):
        return self.conv(x)
    
class CrossAttention(nn.Module):
    def __init__(self,image_dim,text_dim,num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads

        assert image_dim % num_heads == 0

        self.to_q = nn.Linear(image_dim,image_dim)
        self.to_k = nn.Linear(text_dim,image_dim)
        self.to_v = nn.Linear(text_dim,image_dim)
        self.proj = nn.Linear(image_dim,image_dim)

    def forward(self, x, text):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(text)
        v = self.to_v(text)

        q = q.view(B,N,self.num_heads,self.head_dim).transpose(1, 2)
        k = k.view(B,text.shape[1],self.num_heads,self.head_dim).transpose(1, 2)
        v = v.view(B,text.shape[1],self.num_heads,self.head_dim).transpose(1, 2)

        attention = F.scaled_dot_product_attention(q,k,v)
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(B,N,C)

        return self.proj(attention)
    
class Bottleneck(nn.Module):
    def __init__(self,channels,time_dim,text_dim):
        super().__init__()

        self.resblock = ResBlock(channels,channels,time_dim)
        self.norm = nn.LayerNorm(channels)
        self.cross_attention = CrossAttention(image_dim=channels,text_dim=text_dim,num_heads=8)

    def forward(self,x,t,text):
        x = self.resblock(x, t)
        B, C, H, W = x.shape

        x_flat = x.permute(0, 2, 3, 1).reshape(B,H * W,C)
        attention = self.cross_attention(self.norm(x_flat),text)
        x_flat = x_flat + attention
        x = x_flat.reshape(B,H,W,C).permute(0, 3, 1, 2)
        return x
    
class UNet(nn.Module):
    def __init__(self,time_dim=256,text_dim=512):
        super().__init__()
        self.time_sinusoidal = SinusoidalTimeEmbedding(time_dim)
        self.time_embedding = TimeEmbedding(time_dim,time_dim)
        self.input_conv = nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.down_block1 = ResBlock(64,64,time_dim)
        self.downsample1 = Downsample(64,128)
        self.down_block2 = ResBlock(128,128,time_dim)
        self.downsample2 = Downsample(128,256)
        self.bottleneck = Bottleneck(256,time_dim,text_dim)
        self.upsample1 = Upsample(256,128)
        self.up_block1 = ResBlock(256,128,time_dim)
        self.upsample2 = Upsample(128,64)
        self.up_block2 = ResBlock(128,64,time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,3,kernel_size=3,padding=1))
        
    def forward(self,x,t,text):
        t = self.time_sinusoidal(t)
        t = self.time_embedding(t)
        x = self.input_conv(x)
        d1 = self.down_block1(x,t)
        x = self.downsample1(d1)
        d2 = self.down_block2(x,t)
            
        x = self.downsample2(d2)
        x = self.bottleneck(x,t,text)
        x = self.upsample1(x)
        x = torch.cat([x, d2],dim=1)
        x = self.up_block1(x,t)
        x = self.upsample2(x)
        x = torch.cat([x, d1],dim=1)
        x = self.up_block2(x,t)
        return self.output(x)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(time_dim=256,text_dim=512).to(device)
    noisy_images = torch.randn(4,3,64,64).to(device)
    timesteps = torch.tensor([100, 250, 500, 750],device=device)
    text = torch.randn(4,64,512).to(device)
    output = model(noisy_images,timesteps,text)

    print("Input :", noisy_images.shape)
    print("Text  :", text.shape)
    print("Output:", output.shape)
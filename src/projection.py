import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from src.layers import PositionNet
import torchvision.ops as ops


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,groups=1,bias=True):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        self.deform_conv = ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        output = self.deform_conv(x, offset)
        return output
    

        
class FrequencyProtos(nn.Module):
    def __init__(self,
                in_chans,
                high_pass=True,
                ):
        super().__init__()
        self.high_pass = high_pass
        self.in_channels = in_chans
    #    self.freq_deform_conv = DeformableConv2d(in_channels=in_channels, out_channels = in_channels)
        self.freq_weight = nn.Conv2d(in_channels = in_chans,
                                             out_channels=1,
                                              stride=1,
                                              kernel_size=3,
                                              padding=1,
                                              bias=True)
      
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.in_channels // 16, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.in_channels // 16, self.in_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
     
        self.act = "sigmoid"
        self.freq_thres = 0.35
        
    def forward(self, x):
        _,_,h,w = x.shape
   #     x = self.freq_deform_conv(x)
        freq_weight = self.freq_weight(x)
        
        if self.act == "sigmoid":
            freq_weight = freq_weight.sigmoid()
            
        elif self.act == "softmax":
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
            
        else:
            raise NotImplementedError
      
        x_fft = torch.fft.fftshift(torch.fft.fft2(x))
       
        if not self.high_pass:
            low_mask = torch.zeros_like(x_fft, device=x_fft.device)
            low_mask[:,:,round(h/2 - h * self.freq_thres):round(h / 2 + h * self.freq_thres), round(w / 2 - w * self.freq_thres):round(w / 2 + w * self.freq_thres)] = 1.0
            low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * low_mask)).real
            low_x_fft = x_fft * low_mask
            low_c_att = torch.sqrt(self.channel_att(low_x_fft.real) ** 2+self.channel_att(low_x_fft.imag) ** 2 + 1e-8)
            low_part = low_part * freq_weight[:,0:1,] * low_c_att
            
            return low_part
            
        elif self.high_pass:
            high_mask = torch.ones_like(x_fft, device=x_fft.device)
            high_mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h*self.freq_thres),round(w/2-w*self.freq_thres):round(w/2 + w * self.freq_thres)] = 0.0
            high_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * high_mask)).real
            high_x_fft = x_fft * high_mask
            high_c_att = torch.sqrt(self.channel_att(high_x_fft.real) ** 2+self.channel_att(high_x_fft.imag) ** 2 + 1e-8)
            high_part = high_part * freq_weight[:, 0:1,] * high_c_att
            return high_part
            
class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    
class BottleneckDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, high=True, resolution=16):
        super(BottleneckDownsample, self).__init__()
        # 第一个 1x1 卷积，减少通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        # 3x3 卷积，提取空间特征，同时下采样
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)
        if high:
            self.freq = FrequencyProtos(in_chans = out_channels // 4, high_pass=True)
        else:
            self.freq = FrequencyProtos(in_chans = out_channels // 4, high_pass=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        
        # 第二个 1x1 卷积，恢复通道数
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # 下采样捷径分支
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.freq(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 捷径分支
        identity = self.downsample(identity)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

class ResConvNets(nn.Module):
    def __init__(self,high=True):
        super(ResConvNets, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.high = high
        # Bottleneck 模块
        self.layer1 = self._make_layer(64, 256, stride=1, resolution=64)
        self.layer2 = self._make_layer(256, 512, stride=2, resolution=64)
        self.layer3 = self._make_layer(512, 1024, stride=2, resolution=32)

    def _make_layer(self, in_channels, out_channels, stride, resolution):
        layers = []
        # 第一个 Bottleneck 模块进行下采样
        layers.append(BottleneckDownsample(in_channels, out_channels, stride,high=self.high, resolution=resolution))
        # 后续 Bottleneck 模块保持分辨率
        layers.append(BottleneckDownsample(out_channels, out_channels, stride=1, high=self.high, resolution= resolution // stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积和下采样
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过 Bottleneck 模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        return x
    

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0):
        super().__init__() 
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                input,
                mask = None,):
        h = self.norm1(input)
        h1 = self.self_attn(query=h, key=h, value=h, attn_mask=mask)[0]
        h = h + self.dropout(h1)
        h = self.norm2(h)
        return h
        

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b,  l, -1)

        return self.to_out(out)

class CrossAttentionLayer(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.perceiver_fg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.perceiver_bg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
    def forward(self, x, latents):
        x_fg, x_bg = x
        latents_fg, latents_bg = latents
        out_fg = self.perceiver_fg(x_fg, latents_fg)
        out_bg = self.perceiver_bg(x_bg, latents_bg)
        return out_fg,  out_bg


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)
        
        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class SerialSampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        num_queries=[8, 8, 8],
        embedding_dim=768,
        output_dim=1024,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.hf_resampler = Resampler(dim=dim, depth=depth, heads=dim // dim_head, dim_head=dim_head,
                                      num_queries=num_queries[0], embedding_dim=embedding_dim,
                                      output_dim=output_dim, **kwargs)
        self.lf_resampler = Resampler(dim=dim, depth=depth, heads=dim // dim_head, dim_head=dim_head,
                                      num_queries=num_queries[1], embedding_dim=embedding_dim,
                                      output_dim=output_dim, **kwargs)
        self.point_net = PositionNet(in_dim=output_dim, out_dim=output_dim)
  

        self.hf_encoder = ResConvNets(high=True)
        self.lf_encoder = ResConvNets(high=False)
        
    def forward(self, x_objs, obboxes, x_bg):
        x_objs = self.hf_encoder(x_objs)
        x_bg = self.lf_encoder(x_bg)
     
        B = x_bg.shape[0]
        obboxes = torch.from_numpy(np.array([obbox[::2] + obbox[1::2] for obbox in obboxes[0]])).float().to(x_objs.device)
        embed_obboxes = self.point_net(obboxes).unsqueeze(1)
        embed_objs = self.hf_resampler(x_objs) + embed_obboxes
        embed_context = self.lf_resampler(x_bg)
        return embed_objs, embed_context
    
def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)

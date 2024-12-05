import torch
import torch.nn as nn
import torch.nn.functional as F

# 保持原有的MDTA不变
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

# 保持原有的GDFN不变
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                             groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

# 保持原有的TransformerBlock不变
class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                        .contiguous().reshape(b, c, h, w))
        return x

class LiteRestormerGenerator(nn.Module):
    """Lightweight Restormer-based generator with simplified encoder-decoder architecture"""
    
    def __init__(self, input_nc, output_nc, ngf=48, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(LiteRestormerGenerator, self).__init__()
        
        # Architecture parameters
        expansion_factor = 2.66
        num_heads = 4  # 固定使用4个头
        
        # Input convolution
        self.embed_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False)
        )

        # Encoder blocks - 两次下采样
        self.encoder = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
            TransformerBlock(ngf*2, num_heads, expansion_factor),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False),
            TransformerBlock(ngf*4, num_heads, expansion_factor)
        )
        
        # Middle Transform blocks
        self.transform_blocks = nn.Sequential(*[
            TransformerBlock(ngf*4, num_heads, expansion_factor)
            for _ in range(n_blocks)
        ])
        
        # Decoder blocks - 两次上采样
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            TransformerBlock(ngf*2, num_heads, expansion_factor),
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            TransformerBlock(ngf, num_heads, expansion_factor)
        )
        
        # Output convolution
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # Initial embedding
        x = self.embed_conv(x)
        
        # Encoder
        x = self.encoder(x)
        
        # Transform blocks
        x = self.transform_blocks(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Output
        x = self.output(x)
        
        return x
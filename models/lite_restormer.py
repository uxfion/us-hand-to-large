import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

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

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class HybridRestormerGenerator(nn.Module):
    """Hybrid generator combining convolution and Restormer blocks for ultrasound super-resolution"""
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(HybridRestormerGenerator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.expansion_factor = 2.66
        n_downsampling = 2

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling path: PixelUnshuffle -> RestormerBlock -> PixelUnshuffle -> RestormerBlock
        for i in range(n_downsampling):
            mult = 2 ** i
            in_channels = ngf * mult
            
            # PixelUnshuffle下采样
            model += [
                nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelUnshuffle(2)  # 通道数会×4，空间尺寸会/2
            ]
            
            # 每次下采样后添加一个Restormer block
            # 注意：通道数已经变为 in_channels * 2 (因为PixelUnshuffle)
            model += [
                TransformerBlock(
                    channels=in_channels*2,  # PixelUnshuffle后的通道数
                    num_heads=min(2**(i+1), 8),  # 随着层数增加而增加头数，限制最大头数为8
                    expansion_factor=self.expansion_factor
                )
            ]

        # Middle Restormer blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                TransformerBlock(
                    channels=ngf * mult,
                    num_heads=8,  # 中间层使用最多的头
                    expansion_factor=self.expansion_factor
                )
            ]

        # Upsampling path: RestormerBlock -> PixelShuffle -> RestormerBlock -> PixelShuffle
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            in_channels = ngf * mult
            
            # 每次上采样前添加一个Restormer block
            model += [
                TransformerBlock(
                    channels=in_channels,
                    num_heads=2**(n_downsampling-i),
                    expansion_factor=self.expansion_factor
                )
            ]
            
            # PixelShuffle上采样
            model += [
                nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelShuffle(2)  # 通道数会/4，空间尺寸会×2
            ]

        # Output convolution
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    

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
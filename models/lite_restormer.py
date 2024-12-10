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
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

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
    """Lightweight Restormer for ultrasound image enhancement"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):  # 修改默认ngf为64
        super(LiteRestormerGenerator, self).__init__()
        
        # 配置参数
        num_blocks = [2, 3, 2]  # 每层的TransformerBlock数量
        num_heads = [1, 2, 4]   # 每层的注意力头数
        channels = [ngf, ngf*2, ngf*4]  # 每层的通道数 [64, 128, 256]
        expansion_factor = 2.66  # GDFN扩展因子
        num_refinement = 2      # 精调阶段的TransformerBlock数量

        # 初始特征提取
        self.embed_conv = nn.Conv2d(input_nc, channels[0], kernel_size=3, padding=1, bias=False)

        # Encoder阶段
        self.encoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) 
                          for _ in range(num_tb)]) 
            for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
        ])

        # 下采样模块
        self.downs = nn.ModuleList([
            DownSample(channels[i]) for i in range(len(channels)-1)
        ])

        # 上采样模块
        self.ups = nn.ModuleList([
            UpSample(channels[i]) for i in reversed(range(1, len(channels)))
        ])

        # Decoder阶段 - 修正通道数计算
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[1]*2, channels[1], kernel_size=1, bias=False),  # 256->128
                *[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                  for _ in range(num_blocks[1])]
            ),
            nn.Sequential(
                nn.Conv2d(channels[0]*2, channels[0], kernel_size=1, bias=False),  # 128->64
                *[TransformerBlock(channels[0], num_heads[0], expansion_factor)
                  for _ in range(num_blocks[0])]
            )
        ])

        # 精调阶段
        self.refinement = nn.Sequential(*[
            TransformerBlock(channels[0], num_heads[0], expansion_factor)
            for _ in range(num_refinement)
        ])

        # 输出层
        self.output = nn.Conv2d(channels[0], output_nc, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # 初始特征提取
        fo = self.embed_conv(x)
        # print(f"After embed_conv: {fo.shape}")
        
        # Encoder路径
        out_enc1 = self.encoders[0](fo)
        # print(f"After encoder[0]: {out_enc1.shape}")
        
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        # print(f"After encoder[1]: {out_enc2.shape}")
        
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        # print(f"After encoder[2]: {out_enc3.shape}")

        # Decoder路径
        # 第一个decode块：从最深层往回走
        ups0_out = self.ups[0](out_enc3)
        # print(f"After ups[0]: {ups0_out.shape}")
        # print(f"out_enc2 shape: {out_enc2.shape}")
        
        # 特征融合并处理
        out_dec2 = self.decoders[0](torch.cat([ups0_out, out_enc2], dim=1))
        # print(f"After decoder[0]: {out_dec2.shape}")

        # 第二个decode块
        ups1_out = self.ups[1](out_dec2)
        # print(f"After ups[1]: {ups1_out.shape}")
        # print(f"out_enc1 shape: {out_enc1.shape}")
        
        # 特征融合并处理
        out_dec1 = self.decoders[1](torch.cat([ups1_out, out_enc1], dim=1))
        # print(f"After decoder[1]: {out_dec1.shape}")

        # 精调和输出
        fr = self.refinement(out_dec1)
        # print(f"After refinement: {fr.shape}")
        
        out = self.output(fr) + x
        # print(f"Final output: {out.shape}")
        
        return out
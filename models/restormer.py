import torch
import torch.nn as nn
import torch.nn.functional as F

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

class RestormerGenerator(nn.Module):
    """Restormer-based generator adapted to match ResnetGenerator interface"""
    
    def __init__(self, input_nc, output_nc, ngf=48, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Constructor for RestormerGenerator
        
        Parameters:
            input_nc (int)      -- number of channels in input images
            output_nc (int)     -- number of channels in output images
            ngf (int)           -- number of filters in first conv layer (base channel count)
            norm_layer          -- normalization layer (not used, kept for interface compatibility)
            use_dropout (bool)  -- whether to use dropout (not used, kept for interface compatibility)
            n_blocks (int)      -- number of transformer blocks per level
            padding_type (str)  -- padding type (not used, kept for interface compatibility)
        """
        super(RestormerGenerator, self).__init__()
        
        # Configure architecture parameters
        self.num_blocks = [n_blocks, n_blocks, n_blocks, n_blocks]  # blocks per level
        self.num_heads = [1, 2, 4, 8]  # heads per level
        self.channels = [ngf, ngf*2, ngf*4, ngf*8]  # channels per level
        self.expansion_factor = 2.66
        self.num_refinement = 4
        
        # Input convolution
        self.embed_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, self.channels[0], kernel_size=7, padding=0, bias=False)
        )

        # Encoder
        self.encoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(num_ch, num_ah, self.expansion_factor) 
                          for _ in range(num_tb)]) 
            for num_tb, num_ah, num_ch in zip(self.num_blocks, self.num_heads, self.channels)
        ])
        
        # Downsample & Upsample
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in self.channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in reversed(self.channels)[:-1]])
        
        # Reduce blocks
        self.reduces = nn.ModuleList([
            nn.Conv2d(self.channels[i], self.channels[i-1], kernel_size=1, bias=False)
            for i in reversed(range(2, len(self.channels)))
        ])
        
        # Decoder
        self.decoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(self.channels[2], self.num_heads[2], self.expansion_factor)
                          for _ in range(self.num_blocks[2])])
        ])
        self.decoders.append(
            nn.Sequential(*[TransformerBlock(self.channels[1], self.num_heads[1], self.expansion_factor)
                          for _ in range(self.num_blocks[1])])
        )
        self.decoders.append(
            nn.Sequential(*[TransformerBlock(self.channels[1], self.num_heads[0], self.expansion_factor)
                          for _ in range(self.num_blocks[0])])
        )
        
        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(self.channels[1], self.num_heads[0], self.expansion_factor)
            for _ in range(self.num_refinement)
        ])
        
        # Output convolution
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.channels[1], output_nc, kernel_size=7, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        """Standard forward"""
        # Initial embedding
        fo = self.embed_conv(x)
        
        # Encoder path
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))
        
        # Decoder path with skip connections
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        
        # Refinement and output
        fr = self.refinement(fd)
        out = self.output(fr)
        
        return out
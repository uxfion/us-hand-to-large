import torch
import torch.nn as nn
from torch.nn import functional as F
import functools

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            # EMA更新
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()  # commit loss
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class VQResnetGenerator(nn.Module):
    """VQ-ResNet-based generator combining ResNet structure with Vector Quantization
    
    Architecture:
    - Initial processing (256x256)
    - Downsample to 128x128 + VQ1
    - Downsample to 64x64 + VQ2
    - Upsample back to final output
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6, padding_type='reflect',
                 n_embed=512, embed_dim=64, decay=0.99):
        """Initialize VQResnetGenerator
        
        Parameters:
            input_nc (int)      -- number of input image channels
            output_nc (int)     -- number of output image channels
            ngf (int)           -- number of filters in the first conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- whether to use dropout layers
            n_blocks (int)      -- number of ResNet blocks per stage
            padding_type (str)  -- padding type for conv layers
            n_embed (int)       -- number of embeddings in VQ codebook
            embed_dim (int)     -- dimension of VQ embeddings
            decay (float)       -- decay rate for EMA updates in VQ
        """
        super(VQResnetGenerator, self).__init__()
        assert n_blocks >= 0
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Initial convolution block (256x256 -> 256x256)
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # First downsampling + ResBlocks (256x256 -> 128x128)
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # First ResBlock group at 128x128
        self.res_blocks1 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks1.append(
                ResnetBlock(ngf * 2, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # First VQ module at 128x128
        self.vq1_prep = nn.Conv2d(ngf * 2, embed_dim, 1)
        self.vq1 = Quantize(embed_dim, n_embed, decay)
        self.vq1_post = nn.Conv2d(embed_dim, ngf * 2, 1)

        # Second downsampling (128x128 -> 64x64)
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        # Second ResBlock group at 64x64
        self.res_blocks2 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks2.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # Second VQ module at 64x64
        self.vq2_prep = nn.Conv2d(ngf * 4, embed_dim, 1)
        self.vq2 = Quantize(embed_dim, n_embed, decay)
        self.vq2_post = nn.Conv2d(embed_dim, ngf * 4, 1)

        # Third ResBlock group at 64x64
        self.res_blocks3 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks3.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # First upsampling (64x64 -> 128x128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                              padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # Fourth ResBlock group at 128x128
        self.res_blocks4 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks4.append(
                ResnetBlock(ngf * 2, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # Second upsampling (128x128 -> 256x256)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                              padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # Final output convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

        self.vq1_match = nn.Conv2d(embed_dim, ngf * 2, 1)  # 64 -> 128
        self.vq2_match = nn.Conv2d(embed_dim, ngf * 4, 1)  # 64 -> 256 (如果需要)

    def forward(self, x):
        """Forward function
        
        Carefully handle skip connections and feature dimensions
        """
        # Initial convolution
        x = self.init_conv(x)  # 256x256
        
        # First downsample
        x = self.down1(x)  # 128x128
        
        # First ResBlock group
        x_128 = x  # Store for skip connection
        for res_block in self.res_blocks1:
            x = res_block(x)
            
        # First VQ
        quant1 = self.vq1_prep(x)
        quant1, diff1, _ = self.vq1(quant1.permute(0, 2, 3, 1))
        quant1 = quant1.permute(0, 3, 1, 2)
        quant1 = self.vq1_match(quant1)  # 调整通道数
        x = x + self.vq1_post(quant1)  # Skip connection around VQ
        
        # Second downsample
        x = self.down2(x)  # 64x64
        
        # Second ResBlock group
        for res_block in self.res_blocks2:
            x = res_block(x)
            
        # Second VQ
        quant2 = self.vq2_prep(x)
        quant2, diff2, _ = self.vq2(quant2.permute(0, 2, 3, 1))
        quant2 = quant2.permute(0, 3, 1, 2)
        quant2 = self.vq2_match(quant2)  # 调整通道数
        x = x + self.vq2_post(quant2)  # Skip connection around VQ
        
        # Third ResBlock group
        x_64 = x  # Store for skip connection
        for res_block in self.res_blocks3:
            x = res_block(x)
            
        # First upsample
        x = self.up1(x)  # 128x128
        x = x + x_128  # Skip connection from first level
        
        # Fourth ResBlock group
        quant1_matched = self.vq1_match(quant1.detach())  # 确保通道数匹配
        x = x + quant1_matched  # Skip connection from VQ1
        # x = x + quant1.detach()
        for res_block in self.res_blocks4:
            x = res_block(x)
            
        # Second upsample
        x = self.up2(x)  # 256x256
        
        # Final convolution
        x = self.final(x)
        
        return x, diff1 + diff2  # Return both output and VQ loss

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
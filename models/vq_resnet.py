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
    """优化版VQ-ResNet-based生成器，对输入尺寸不敏感
    
    改进:
    - 使用连接(concatenation)代替加法(addition)
    - 使用更灵活的上采样方法
    - 解码量化特征后再与上采样特征连接
    - 所有卷积层预定义在__init__中
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6, padding_type='reflect',
                 n_embed=512, embed_dim=64, decay=0.99):
        """初始化VQResnetGenerator
        
        Parameters:
            input_nc (int)      -- 输入图像通道数
            output_nc (int)     -- 输出图像通道数
            ngf (int)           -- 第一个卷积层的过滤器数量
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 是否使用dropout层
            n_blocks (int)      -- 每个阶段的ResNet块数量
            padding_type (str)  -- 卷积层的填充类型
            n_embed (int)       -- VQ码本中的嵌入数量
            embed_dim (int)     -- VQ嵌入的维度
            decay (float)       -- VQ中EMA更新的衰减率
        """
        super(VQResnetGenerator, self).__init__()
        assert n_blocks >= 0
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 初始卷积块
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 第一个下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # 第一个ResBlock组 at 128x128
        self.res_blocks1 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks1.append(
                ResnetBlock(ngf * 2, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # 第一个VQ模块
        self.vq1_prep = nn.Conv2d(ngf * 2, embed_dim, 1)
        self.vq1 = Quantize(embed_dim, n_embed, decay)
        self.vq1_post = nn.Sequential(
            nn.Conv2d(embed_dim, ngf * 2, 1),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # 第二个下采样
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        # 第二个ResBlock组
        self.res_blocks2 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks2.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # 第二个VQ模块
        self.vq2_prep = nn.Conv2d(ngf * 4, embed_dim, 1)
        self.vq2 = Quantize(embed_dim, n_embed, decay)
        self.vq2_post = nn.Sequential(
            nn.Conv2d(embed_dim, ngf * 4, 1),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        # VQ2特征融合层 - 预定义
        self.vq2_fusion = nn.Sequential(
            nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        # 第三个ResBlock组
        self.res_blocks3 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks3.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # 第一次上采样
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # 跳跃连接融合层
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # VQ1特征融合层
        self.vq1_fusion = nn.Sequential(
            nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # 第四个ResBlock组
        self.res_blocks4 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks4.append(
                ResnetBlock(ngf * 2, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # 第二次上采样
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 最终输出卷积
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def _resize_if_needed(self, x, target):
        """内部辅助方法：在需要时调整特征图尺寸"""
        if x.size(2) != target.size(2) or x.size(3) != target.size(3):
            return F.interpolate(x, size=(target.size(2), target.size(3)), 
                               mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        """前向函数"""
        # 初始卷积
        x = self.init_conv(x)
        
        # 第一次下采样
        x = self.down1(x)
        
        # 第一个ResBlock组
        x_128 = x  # 存储用于跳跃连接
        for res_block in self.res_blocks1:
            x = res_block(x)
            
        # 第一个VQ
        quant1 = self.vq1_prep(x)
        quant1, diff1, _ = self.vq1(quant1.permute(0, 2, 3, 1))
        quant1 = quant1.permute(0, 3, 1, 2)
        quant1_decoded = self.vq1_post(quant1)
        
        # 第二次下采样
        x = self.down2(x)
        
        # 第二个ResBlock组
        for res_block in self.res_blocks2:
            x = res_block(x)
            
        # 第二个VQ
        quant2 = self.vq2_prep(x)
        quant2, diff2, _ = self.vq2(quant2.permute(0, 2, 3, 1))
        quant2 = quant2.permute(0, 3, 1, 2)
        quant2_decoded = self.vq2_post(quant2)
        
        # 将解码的VQ2与主路径特征连接并融合
        x = torch.cat([x, quant2_decoded], dim=1)
        x = self.vq2_fusion(x)
        
        # 第三个ResBlock组
        x_64 = x  # 存储用于潜在跳跃连接（如果需要）
        for res_block in self.res_blocks3:
            x = res_block(x)
            
        # 第一次上采样
        x = self.up1(x)
        
        # 与第一层特征通过跳跃连接融合
        x_128_resized = self._resize_if_needed(x_128, x)
        x = torch.cat([x, x_128_resized], dim=1)
        x = self.skip_fusion(x)
        
        # 连接并融合解码的VQ1特征
        quant1_resized = self._resize_if_needed(quant1_decoded, x)
        x = torch.cat([x, quant1_resized], dim=1)
        x = self.vq1_fusion(x)
        
        # 第四个ResBlock组
        for res_block in self.res_blocks4:
            x = res_block(x)
            
        # 第二次上采样
        x = self.up2(x)
        
        # 最终卷积
        x = self.final(x)
        
        return x, diff1 + diff2  # 返回输出和VQ损失

class ResnetBlock(nn.Module):
    """定义一个Resnet块"""

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
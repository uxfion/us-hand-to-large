import torch
import torch.nn as nn
import functools
from .quantize import EMAVectorQuantizer
from .contmix import ContMixBlock, LayerNorm2d  # 导入ContMix模块


class ContmixVQDualEnDecoderGenerator(nn.Module):
    """
    双编解码器VQ生成器，用于掌上超声和大型超声图像的相互转换
    支持四种数据流：
    1. A->A (掌超重建)
    2. B->B (大型重建)  
    3. A->B (掌超转大型)
    4. B->A (大型转掌超)
    """
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=9, padding_type='reflect',
                 n_embed=512, embed_dim=256, beta=0.25, decay=0.99,
                 use_contmix=True):  # 添加ContMix开关
        """
        参数:
            input_nc (int) -- 输入图像通道数
            output_nc (int) -- 输出图像通道数
            ngf (int) -- 第一层卷积的滤波器数量
            norm_layer -- 归一化层
            use_dropout (bool) -- 是否使用dropout
            n_blocks (int) -- ResNet块的数量
            padding_type (str) -- 填充类型
            n_embed (int) -- VQ码本大小
            embed_dim (int) -- VQ嵌入维度
            beta (float) -- VQ commitment loss权重
            decay (float) -- EMA衰减率
            use_contmix (bool) -- 是否使用ContMix块增强特征提取
        """
        super(ContmixVQDualEnDecoderGenerator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.use_contmix = use_contmix
            
        # 编码器A（掌上超声）
        self.encoder_A = self._build_encoder(input_nc, ngf, norm_layer, use_bias, 
                                            n_blocks, padding_type, use_dropout)
        
        # 编码器B（大型超声）
        self.encoder_B = self._build_encoder(input_nc, ngf, norm_layer, use_bias, 
                                            n_blocks, padding_type, use_dropout)
        
        # 共享VQ层
        # 计算编码器输出的通道数（经过2次下采样，通道数变为ngf * 4）
        encoder_out_channels = ngf * 4
        self.vq_layer = EMAVectorQuantizer(
            n_embed=n_embed,
            embedding_dim=embed_dim,
            beta=beta,
            decay=decay
        )
        
        # 投影层：将编码器输出映射到VQ嵌入维度
        self.encoder_proj = nn.Conv2d(encoder_out_channels, embed_dim, 1)
        self.decoder_proj = nn.Conv2d(embed_dim, encoder_out_channels, 1)
        
        # 解码器A（生成掌上超声）
        self.decoder_A = self._build_decoder(encoder_out_channels, output_nc, ngf, norm_layer, use_bias)

        # 解码器B（生成大型超声）
        self.decoder_B = self._build_decoder(encoder_out_channels, output_nc, ngf, norm_layer, use_bias)
        
    def _build_encoder(self, input_nc, ngf, norm_layer, use_bias, n_blocks, padding_type, use_dropout):
        """构建编码器"""
        encoder = []
        
        # 初始卷积层
        encoder += [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                   norm_layer(ngf),
                   nn.ReLU(True)]
        
        # 下采样层
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       nn.ReLU(True)]
        
        # 核心特征提取块（在64×64分辨率）
        mult = 2 ** n_downsampling  # = 4, 通道数256
        
        if self.use_contmix:
            # 渐进式混合策略：3个阶段，每阶段3个块
            
            # Stage 1: 前3个ResNet块 - 快速局部特征提取
            for i in range(3):
                encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, 
                                       norm_layer=norm_layer, use_dropout=use_dropout, 
                                       use_bias=use_bias)]
            


            encoder += [ContMixBlock(
                dim=ngf * mult,  # 256
                kernel_size=7,  # 中等核，适合中程伪影
                smk_size=3,      # 辅助小核捕获局部噪声
                num_heads=2,     # 平衡的注意力头数
                mlp_ratio=3,     # 适中的MLP扩展
                res_scale=True,  # 使用残差缩放稳定训练
                ls_init_value=1.0,  # 参考OverLoCK的设置
                drop_path=0,  # 渐进式dropout
                norm_layer=LayerNorm2d,  # ContMix标准配置
                use_gemm=True,   # 启用高效实现
                deploy=False     # 训练模式
            )]
            


            encoder += [ContMixBlock(
                dim=ngf * mult,
                kernel_size=13,
                smk_size=5,
                num_heads=4,
                mlp_ratio=3,
                res_scale=True,
                ls_init_value=1.0,
                drop_path=0,
                norm_layer=LayerNorm2d,
                use_gemm=True,
                deploy=False
            )]

            for i in range(3):
                encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, 
                                       norm_layer=norm_layer, use_dropout=use_dropout, 
                                       use_bias=use_bias)]

        else:
            # 原始版本：9个ResNet块
            for i in range(n_blocks):
                encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, 
                                       norm_layer=norm_layer, use_dropout=use_dropout, 
                                       use_bias=use_bias)]
            
        return nn.Sequential(*encoder)
    
    def _build_decoder(self, input_channels, output_nc, ngf, norm_layer, use_bias):
        """构建解码器"""
        decoder = []
        
        # 上采样层
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(input_channels if i == 0 else ngf * mult, 
                                          int(ngf * mult / 2),
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1,
                                          bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        
        # 输出层
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]
        
        return nn.Sequential(*decoder)
    
    def encode(self, x, domain='A'):
        """编码输入图像"""
        if domain == 'A':
            features = self.encoder_A(x)
        else:
            features = self.encoder_B(x)
        
        # 投影到VQ嵌入空间
        features = self.encoder_proj(features)
        return features
    
    def decode(self, features, domain='A'):
        """解码特征到图像"""
        # 从VQ嵌入空间投影回解码器输入空间
        features = self.decoder_proj(features)
        
        if domain == 'A':
            output = self.decoder_A(features)
        else:
            output = self.decoder_B(features)
        return output
    
    def forward(self, x, direction='AtoB'):
        """
        前向传播
        参数:
            x: 输入图像
            direction: 转换方向，可选值：
                'AtoB': 掌超转大型
                'BtoA': 大型转掌超
                'AtoA': 掌超重建
                'BtoB': 大型重建
        """
        # 解析方向
        source_domain = direction[0]
        target_domain = direction[-1]
        
        # 编码
        features = self.encode(x, source_domain)
        
        # 向量量化
        quantized, vq_loss, (perplexity, _, indices) = self.vq_layer(features)

        # 解码
        output = self.decode(quantized, target_domain)

        # TODO: if direction == 'BtoB':

        # 累积VQ损失（用于多次前向传播）
        if not hasattr(self, 'accumulated_vq_loss'):
            self.accumulated_vq_loss = 0
        self.accumulated_vq_loss = self.accumulated_vq_loss + vq_loss
        
        # 保存VQ相关信息供后续使用
        # TODO: 码本利用率
        self.vq_loss = vq_loss
        self.perplexity = perplexity
        self.indices = indices
        
        return output
    
    def get_vq_loss(self):
        """获取累积的VQ损失并重置"""
        if hasattr(self, 'accumulated_vq_loss'):
            vq_loss = self.accumulated_vq_loss
            self.accumulated_vq_loss = 0  # 重置累积损失
            return vq_loss
        return 0
    
    def get_codebook_usage(self):
        """获取码本使用情况统计"""
        if hasattr(self, 'indices') and self.indices is not None:
            # 统计每个码本向量的使用频率
            usage = torch.bincount(self.indices.view(-1), minlength=self.vq_layer.n_embed)
            usage_rate = (usage > 0).float().mean()
            return usage, usage_rate
        return None, 0.0


class ResnetBlock(nn.Module):
    """定义ResNet块"""
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化ResNet块"""
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """构建卷积块"""
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                      norm_layer(dim), 
                      nn.ReLU(True)]
        
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
            
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                      norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """前向传播（带跳跃连接）"""
        out = x + self.conv_block(x)
        return out
import torch
import torch.nn as nn
from torch.nn import functional as F
import functools

class VectorQuantizer(nn.Module):
    """基于Taming Transformers中VQGAN的向量量化器
    
    改进:
    - 使用更高效的距离计算
    - 双向commitment loss
    - 优化的EMA更新策略
    - 使用Laplace分布初始化码本
    """
    def __init__(self, dim, n_embed, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.beta = beta  # commitment loss权重参数
        self.decay = decay
        self.eps = eps

        # 较好的码本初始化 
        self.embedding = nn.Embedding(n_embed, dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        
        # 注册EMA更新所需的缓冲区
        self.register_buffer("ema_cluster_size", torch.zeros(n_embed))
        self.register_buffer("ema_weight", torch.zeros_like(self.embedding.weight))
        self.register_buffer("ema_inited", torch.zeros(1, dtype=torch.bool))

    def forward(self, z):
        # 将输入展平为二维张量
        z_flattened = z.reshape(-1, self.dim)
        
        # 计算欧氏距离（更高效的实现）
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        # 找到最近的码本向量
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # 计算perplexity (Taming Transformers中使用)
        min_encodings = F.one_hot(min_encoding_indices, self.n_embed).float()
        perplexity = torch.exp(-torch.sum(min_encodings * torch.log(min_encodings.clamp(min=1e-10)), dim=-1)).mean()
        
        # 计算码本使用情况统计
        encodings_sum = torch.sum(min_encodings, dim=0)
        usage_ratio = torch.sum(encodings_sum > 0) / self.n_embed  # 使用了的码本比例
        
        # VQ Losses: commitment loss和codebook loss
        # 双向设计可以更好地稳定训练
        q_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach()) 
        vq_loss = q_loss + self.beta * commitment_loss
        
        # Straight-Through估计器
        z_q = z + (z_q - z).detach()
        
        # 仅在训练阶段更新码本
        if self.training:
            # EMA更新码本
            if not self.ema_inited:
                self.ema_cluster_size.data.copy_(min_encodings.sum(0) + self.eps)
                # Fix: Change order of matrix multiplication
                self.ema_weight.data.copy_(torch.matmul(min_encodings.t(), z_flattened))
                self.ema_inited.data.copy_(torch.ones(1, dtype=torch.bool))
            else:
                self.ema_cluster_size.data.mul_(self.decay).add_(
                    min_encodings.sum(0), alpha=1-self.decay)
                # Fix: Change order of matrix multiplication
                self.ema_weight.data.mul_(self.decay).add_(
                    torch.matmul(min_encodings.t(), z_flattened), alpha=1-self.decay)
            
            # 计算归一化权重
            n = self.ema_cluster_size.sum()
            cluster_size = self.ema_cluster_size + self.eps
            dw = cluster_size / (n * cluster_size / torch.sum(cluster_size))
            
            # 更新嵌入权重
            embed_normalized = self.ema_weight / dw.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)
            
        # 返回码本内部统计信息供监控
        stats = {
            'perplexity': perplexity,
            'used_codes': (encodings_sum > 0).sum(),
            'usage_ratio': usage_ratio,
            'codebook_loss': q_loss.detach(),
            'commitment_loss': commitment_loss.detach(),
        }
            
        return z_q, vq_loss, stats

    def get_codebook_similarity(self):
        """返回码本内部的相似度矩阵，用于正则化"""
        w = self.embedding.weight
        w_norm = w / (torch.norm(w, dim=1, keepdim=True) + 1e-8)
        sim = torch.matmul(w_norm, w_norm.t())
        sim = sim - torch.eye(sim.shape[0], device=sim.device)  # 去除自相似
        return sim

    def get_codebook_entropy(self):
        """计算码本的熵，衡量码本使用均匀性"""
        if not torch.all(self.ema_inited):
            return torch.tensor(0.0, device=self.embedding.weight.device)
        
        # 使用码本项的频率作为概率分布
        p = self.ema_cluster_size / (self.ema_cluster_size.sum() + self.eps)
        entropy = -torch.sum(p * torch.log(p + self.eps))
        return entropy

class AttentionBlock(nn.Module):
    """从Taming Transformers中借鉴的自注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # reshape以计算注意力
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h*w)  # b, c, hw
        v = v.reshape(b, c, h*w)  # b, c, hw

        # 注意力计算
        attn = torch.bmm(q, k)  # b, hw, hw
        attn = attn * (c ** -0.5)  # 缩放因子
        attn = F.softmax(attn, dim=2)

        # 应用注意力权重
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, h, w)
        
        return x + self.proj_out(out)

class ResnetBlock(nn.Module):
    """改进的ResNet块，基于Taming Transformers"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.activation = nn.SiLU(inplace=True)  # 使用SiLU代替ReLU
        
        # 第一个卷积层
        padding = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1.append(nn.ReflectionPad2d(1))
        elif padding_type == 'replicate':
            conv1.append(nn.ReplicationPad2d(1))
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError(f'padding type {padding_type} not implemented')
            
        conv1.extend([
            nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
            norm_layer(dim),
            nn.SiLU(inplace=True)
        ])
        
        # 第二个卷积层
        padding = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2.append(nn.ReflectionPad2d(1))
        elif padding_type == 'replicate':
            conv2.append(nn.ReplicationPad2d(1))
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError(f'padding type {padding_type} not implemented')
            
        conv2.append(nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias))
        
        # 是否使用dropout
        if use_dropout:
            conv2.append(nn.Dropout(0.5))
            
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)

    def forward(self, x):
        out = self.activation(self.norm1(x))
        out = self.conv1(out)
        out = self.conv2(out)
        return x + out

class VQAttenResnetGenerator(nn.Module):
    """优化版VQ-ResNet-based生成器，整合了Taming Transformers中VQGAN的创新
    
    改进:
    - 使用VectorQuantizer代替Quantize
    - 添加自注意力机制
    - 优化ResnetBlock设计
    - 使用SiLU激活函数
    - 改进的VQ前后处理
    - 更大的码本和嵌入维度
    - 返回详细监控指标
    """
    def __init__(self, input_nc, output_nc, ngf=96, norm_layer=nn.GroupNorm,
                 use_dropout=False, n_blocks=9, padding_type='reflect',
                 n_embed=1024, embed_dim=256, decay=0.99):
        """初始化VQResnetGenerator
        
        Parameters:
            input_nc (int)      -- 输入图像通道数
            output_nc (int)     -- 输出图像通道数
            ngf (int)           -- 第一个卷积层的过滤器数量 (增大到96)
            norm_layer          -- 标准化层 (默认改为GroupNorm)
            use_dropout (bool)  -- 是否使用dropout层
            n_blocks (int)      -- 每个阶段的ResNet块数量 (增加到9)
            padding_type (str)  -- 卷积层的填充类型
            n_embed (int)       -- VQ码本中的嵌入数量 (增加到1024)
            embed_dim (int)     -- VQ嵌入的维度 (增加到256)
            decay (float)       -- VQ中EMA更新的衰减率
        """
        super(VQAttenResnetGenerator, self).__init__()
        assert n_blocks >= 0
        
        # 确定使用偏置项
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            if norm_layer == nn.GroupNorm:
                norm_layer = lambda num_channels: nn.GroupNorm(32, num_channels)
                use_bias = True
            else:
                use_bias = norm_layer != nn.BatchNorm2d

        # 初始卷积块
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.SiLU(inplace=True)
        )

        # 第一个下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.SiLU(inplace=True)
        )

        # 第一个ResBlock组
        self.res_blocks1 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks1.append(
                ResnetBlock(ngf * 2, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # 第一个VQ模块
        self.vq1_prep = nn.Sequential(
            norm_layer(ngf * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(ngf * 2, embed_dim, 1, bias=True)
        )
        self.vq1 = VectorQuantizer(embed_dim, n_embed, beta=0.25, decay=decay)
        self.vq1_post = nn.Sequential(
            nn.Conv2d(embed_dim, ngf * 2, 1, bias=True),
            norm_layer(ngf * 2),
            nn.SiLU(inplace=True)
        )

        # 第二个下采样
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.SiLU(inplace=True)
        )

        # 第二个ResBlock组
        self.res_blocks2 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks2.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )
        
        # 添加自注意力层
        self.attn = AttentionBlock(ngf * 4)

        # 第二个VQ模块
        self.vq2_prep = nn.Sequential(
            norm_layer(ngf * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(ngf * 4, embed_dim, 1, bias=True)
        )
        self.vq2 = VectorQuantizer(embed_dim, n_embed, beta=0.25, decay=decay)
        self.vq2_post = nn.Sequential(
            nn.Conv2d(embed_dim, ngf * 4, 1, bias=True),
            norm_layer(ngf * 4),
            nn.SiLU(inplace=True)
        )

        # VQ2特征融合层
        self.vq2_fusion = nn.Sequential(
            nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.SiLU(inplace=True)
        )

        # 第三个ResBlock组
        self.res_blocks3 = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks3.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, 
                           norm_layer=norm_layer, use_dropout=use_dropout, 
                           use_bias=use_bias)
            )

        # 第一次上采样（改用nearest模式避免棋盘伪影）
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.SiLU(inplace=True)
        )

        # 跳跃连接融合层
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.SiLU(inplace=True)
        )

        # VQ1特征融合层
        self.vq1_fusion = nn.Sequential(
            nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.SiLU(inplace=True)
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
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.SiLU(inplace=True)
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
                               mode='nearest')
        return x

    def forward(self, x):
        """前向函数，增强版返回包含更多监控信息"""
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
        quant1, loss1, stats1 = self.vq1(quant1)
        quant1_decoded = self.vq1_post(quant1)
        
        # 第二次下采样
        x = self.down2(x)
        
        # 第二个ResBlock组
        for res_block in self.res_blocks2:
            x = res_block(x)
        
        # 应用自注意力
        x = self.attn(x)    
            
        # 第二个VQ
        quant2 = self.vq2_prep(x)
        quant2, loss2, stats2 = self.vq2(quant2)
        quant2_decoded = self.vq2_post(quant2)
        
        # 将解码的VQ2与主路径特征连接并融合
        x = torch.cat([x, quant2_decoded], dim=1)
        x = self.vq2_fusion(x)
        
        # 第三个ResBlock组
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
        
        # 组合VQ损失和统计信息
        vq_loss = loss1 + loss2
        vq_stats = {
            'vq1_perplexity': stats1['perplexity'],
            'vq1_used_codes': stats1['used_codes'],
            'vq1_usage_ratio': stats1['usage_ratio'],
            'vq2_perplexity': stats2['perplexity'],
            'vq2_used_codes': stats2['used_codes'],
            'vq2_usage_ratio': stats2['usage_ratio'],
            'vq1_codebook_loss': stats1['codebook_loss'],
            'vq1_commitment_loss': stats1['commitment_loss'],
            'vq2_codebook_loss': stats2['codebook_loss'],
            'vq2_commitment_loss': stats2['commitment_loss'],
        }
        
        return x, vq_loss, vq_stats
    
    def get_codebook_reg_loss(self):
        """计算码本正则化损失"""
        sim1 = self.vq1.get_codebook_similarity()
        sim2 = self.vq2.get_codebook_similarity()
        # 惩罚码本向量之间的高相似度
        reg_loss = torch.mean(torch.pow(torch.relu(sim1), 2)) + torch.mean(torch.pow(torch.relu(sim2), 2))
        return reg_loss
        
    def get_last_layer(self):
        """获取最后一层权重，用于自适应损失权重计算"""
        return self.final[1].weight
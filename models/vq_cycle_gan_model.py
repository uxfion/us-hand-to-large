import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class VQCycleGANModel(BaseModel):
    """
    VQ-CycleGAN模型，集成了向量量化(VQ)的CycleGAN
    用于掌上超声和大型超声图像之间的转换和增强
    
    该模型同时训练四种任务：
    1. 掌超重建 (A->A)
    2. 大型重建 (B->B)
    3. 掌超转大型 (A->B)
    4. 大型转掌超 (B->A)
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加VQ-CycleGAN特定的选项"""
        parser.set_defaults(no_dropout=True)  # CycleGAN默认不使用dropout
        
        if is_train:
            # CycleGAN原有参数
            parser.add_argument('--lambda_A', type=float, default=10.0, 
                              help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, 
                              help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0,
                              help='use identity mapping. Set to 0 for medical image enhancement.')
            
            # VQ相关参数
            parser.add_argument('--n_embed', type=int, default=512,
                              help='number of embeddings in VQ codebook')
            parser.add_argument('--embed_dim', type=int, default=256,
                              help='dimension of embeddings in VQ codebook')
            parser.add_argument('--beta', type=float, default=0.25,
                              help='commitment cost for VQ')
            parser.add_argument('--decay', type=float, default=0.99,
                              help='decay rate for EMA in VQ')
            
            # 重建损失权重
            parser.add_argument('--lambda_rec_A', type=float, default=10.0,
                              help='weight for A domain reconstruction loss')
            parser.add_argument('--lambda_rec_B', type=float, default=10.0,
                              help='weight for B domain reconstruction loss')
            parser.add_argument('--lambda_vq', type=float, default=1.0,
                              help='weight for VQ loss')
            
            # 配对损失权重
            parser.add_argument('--lambda_paired', type=float, default=10.0,
                              help='weight for paired loss (semi_paired and aligned modes)')
            
        return parser

    def __init__(self, opt):
        """初始化VQ-CycleGAN模型"""
        BaseModel.__init__(self, opt)
        
        # 定义损失名称（用于打印和保存）
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 
                          'D_B', 'G_B', 'cycle_B',
                          'rec_A', 'rec_B', 'vq', 'vq_pp',
                          'codebook_usage', 'avg_usage',  # 简化码本监控
                          'paired']  # 添加配对损失
        
        # 只有在使用identity loss时才添加
        if self.isTrain and self.opt.lambda_identity > 0.0:
            self.loss_names.extend(['idt_A', 'idt_B'])
        
        # 定义要保存/显示的图像
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'recon_A']  # 添加recon_A (A->A重建)
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'recon_B']  # 添加recon_B (B->B重建)
        
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')  # real_B通过AtoB路径的结果
            visual_names_B.append('idt_B')  # real_A通过BtoA路径的结果

        self.visual_names = visual_names_A + visual_names_B
        
        # 定义模型名称
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']  # 只有一个生成器G
        else:
            self.model_names = ['G']

        # 定义网络
        # 使用VQ双编解码器生成器，注意使用框架中的参数名
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
            opt.n_embed, opt.embed_dim, opt.beta, opt.decay
        )
        
        # 为了兼容性，创建别名
        self.netG_A = self.netG
        self.netG_B = self.netG
        
        if self.isTrain:
            # 定义判别器
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, 
                                          opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, 
                                          opt.init_gain, self.gpu_ids)
            
            # 初始化码本监控指标（使用tensor以便在不同设备间传递）
            self.loss_codebook_usage = torch.tensor(0.0, device=self.device)
            self.loss_avg_usage = torch.tensor(0.0, device=self.device)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)
            
            # 图像池
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            # 损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionRec = torch.nn.L1Loss()  # 重建损失
            self.criterionPaired = torch.nn.L1Loss()  # 配对损失
            
            # 优化器
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """解包输入数据"""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mode = input['mode']  # 获取模式（unaligned, aligned, semi_paired）
        
        # 处理batch中的模式信息
        if isinstance(self.mode, (list, tuple)):
            # 如果mode是列表（batch中每个样本的模式）
            self.batch_modes = self.mode
            # 创建配对数据掩码
            self.paired_mask = torch.tensor([
                mode in ['semi_paired', 'aligned'] for mode in self.batch_modes
            ], dtype=torch.bool, device=self.device)
        else:
            # 如果mode是单个字符串（整个batch同一模式）
            self.batch_modes = [self.mode] * self.real_A.size(0)
            self.paired_mask = torch.tensor([
                self.mode in ['semi_paired', 'aligned']
            ] * self.real_A.size(0), dtype=torch.bool, device=self.device)
        
        # 计算配对数据的数量
        self.num_paired = self.paired_mask.sum().item()

    def forward(self):
        """前向传播，计算所有需要的输出"""
        # 跨域转换
        self.fake_B = self.netG(self.real_A, direction='AtoB')  # A -> B
        self.rec_A = self.netG(self.fake_B, direction='BtoA')   # B -> A (循环)
        self.fake_A = self.netG(self.real_B, direction='BtoA')  # B -> A
        self.rec_B = self.netG(self.fake_A, direction='AtoB')   # A -> B (循环)
        
        # 域内重建（VQ-VAE功能）
        self.recon_A = self.netG(self.real_A, direction='AtoA')  # A -> A
        self.recon_B = self.netG(self.real_B, direction='BtoB')  # B -> B

    def backward_D_basic(self, netD, real, fake):
        """计算基本的判别器损失"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """计算判别器D_A的损失"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """计算判别器D_B的损失"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """计算生成器G的损失"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_rec_A = self.opt.lambda_rec_A
        lambda_rec_B = self.opt.lambda_rec_B
        lambda_vq = self.opt.lambda_vq
        lambda_paired = self.opt.lambda_paired
        
        # Identity loss
        if lambda_idt > 0:
            # G_A(real_B)应该保持real_B不变（B已经是高质量了，不需要增强）
            self.idt_A = self.netG(self.real_B, direction='AtoB')  # 注意：是AtoB路径！
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            # G_B(real_A)应该保持real_A不变（A已经是低质量了，不需要退化）
            self.idt_B = self.netG(self.real_A, direction='BtoA')  # 注意：是BtoA路径！
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Reconstruction loss (域内重建)
        self.loss_rec_A = self.criterionRec(self.recon_A, self.real_A) * lambda_rec_A
        self.loss_rec_B = self.criterionRec(self.recon_B, self.real_B) * lambda_rec_B

        # Paired loss (配对数据损失：fake_B应该与real_B匹配)
        if self.num_paired > 0:
            # 只对配对数据计算损失
            paired_fake_B = self.fake_B[self.paired_mask]
            paired_real_B = self.real_B[self.paired_mask]
            
            # 计算配对数据比例并应用线性缩放
            self.paired_ratio = self.num_paired / len(self.batch_modes) if len(self.batch_modes) > 0 else 0.0
            self.loss_paired = self.criterionPaired(paired_fake_B, paired_real_B) * lambda_paired * self.paired_ratio
            
            # # 调试信息（可以在训练稳定后删除）
            # if hasattr(self, 'batch_count') and self.batch_count % 100 == 0:
            #     print(f"Batch {self.batch_count}: {self.num_paired}/{len(self.batch_modes)} paired samples, "
            #           f"paired_ratio: {self.paired_ratio:.3f}, loss_paired: {self.loss_paired.item():.4f}")
        else:
            self.paired_ratio = 0.0
            self.loss_paired = torch.tensor(0.0, device=self.device)
        
        # VQ loss
        # 处理DataParallel的情况
        vq_generator = self.netG.module if hasattr(self.netG, 'module') else self.netG
        self.loss_vq = vq_generator.get_vq_loss() * lambda_vq
        
        # VQ perplexity (用于监控，不参与反向传播)
        if hasattr(vq_generator, 'perplexity') and vq_generator.perplexity is not None:
            self.loss_vq_pp = vq_generator.perplexity.mean()
        else:
            self.loss_vq_pp = torch.tensor(0.0)
        
        # 初始化码本监控指标（如果还没有值）
        if not hasattr(self, 'loss_codebook_usage'):
            self.loss_codebook_usage = torch.tensor(0.0, device=self.device)
        if not hasattr(self, 'loss_avg_usage'):
            self.loss_avg_usage = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        self.loss_G = (self.loss_G_A + self.loss_G_B + 
                      self.loss_cycle_A + self.loss_cycle_B + 
                      self.loss_idt_A + self.loss_idt_B +
                      self.loss_rec_A + self.loss_rec_B +
                      self.loss_vq + self.loss_paired)

        # TODO: if xxx: self.loss_G += 
        
        self.loss_G.backward()

    def optimize_parameters(self):
        """计算损失、梯度，并更新网络权重"""
        # 前向传播
        self.forward()
        
        # 更新生成器G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # 更新判别器D_A和D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        
        # 更新码本使用率监控（不需要每个batch都更新，可以降低频率）
        if hasattr(self, 'batch_count'):
            self.batch_count += 1
        else:
            self.batch_count = 1
            
        if self.batch_count % 100 == 0:  # 每100个batch更新一次
            self.evaluate_codebook()
    
    def get_current_visuals(self):
        """返回当前的可视化图像"""
        visual_ret = super().get_current_visuals()
        
        # 添加VQ码本使用情况的可视化（可选）
        if hasattr(self.netG, 'indices') and self.netG.indices is not None:
            # 可以在这里添加码本使用情况的可视化
            pass
            
        return visual_ret
    
    def get_current_losses(self):
        """返回当前的损失字典，添加配对损失的统计信息"""
        losses_dict = super().get_current_losses()
        
        # 添加配对损失的额外统计信息
        if hasattr(self, 'paired_ratio'):
            losses_dict['paired_ratio'] = float(self.paired_ratio)
        
        return losses_dict
    
    def evaluate_codebook(self):
        """评估VQ码本的使用情况"""
        vq_generator = self.netG.module if hasattr(self.netG, 'module') else self.netG
        usage, usage_rate = vq_generator.get_codebook_usage()
        
        if usage is not None:
            # 确保tensor在正确的设备上
            if not isinstance(usage_rate, torch.Tensor):
                usage_rate = torch.tensor(usage_rate, device=self.device)
            else:
                usage_rate = usage_rate.to(self.device)
            
            # 添加到loss中用于tensorboard/visdom记录
            self.loss_codebook_usage = usage_rate.detach()
            
            # avg_usage需要处理空的情况
            if (usage > 0).any():
                self.loss_avg_usage = usage[usage > 0].float().mean().detach().to(self.device)
            else:
                self.loss_avg_usage = torch.tensor(0.0, device=self.device)
            
        return usage, usage_rate
    

    
    def handle_codebook_collapse(self, threshold=0.5):
        """处理码本崩塌问题"""
        vq_generator = self.netG.module if hasattr(self.netG, 'module') else self.netG
        usage, usage_rate = vq_generator.get_codebook_usage()
        
        if usage_rate < threshold:
            print(f"Warning: Codebook usage rate ({usage_rate:.2%}) below threshold ({threshold:.0%})")
            
            # 可选的恢复策略：
            # 1. 重新初始化未使用的码本向量
            if hasattr(vq_generator.vq_layer, 'embedding'):
                unused_indices = (usage == 0).nonzero().squeeze()
                if len(unused_indices) > 0:
                    # 使用已使用向量的扰动版本重新初始化
                    used_indices = (usage > 0).nonzero().squeeze()
                    if len(used_indices) > 0:
                        # 随机选择一些已使用的向量
                        random_used = used_indices[torch.randint(0, len(used_indices), (len(unused_indices),))]
                        # 添加噪声
                        noise = torch.randn_like(vq_generator.vq_layer.embedding.weight[random_used]) * 0.1
                        vq_generator.vq_layer.embedding.weight.data[unused_indices] = \
                            vq_generator.vq_layer.embedding.weight.data[random_used] + noise
                        print(f"Reinitialized {len(unused_indices)} unused codebook vectors")
            
            # 2. 调整beta值（可选）
            # vq_generator.vq_layer.beta *= 0.9
            
        return usage_rate
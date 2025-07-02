import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .vq_dual_generator import VQDualEnDecoderGenerator


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
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                              help='use identity mapping.')
            
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
            
        return parser

    def __init__(self, opt):
        """初始化VQ-CycleGAN模型"""
        BaseModel.__init__(self, opt)
        
        # 定义损失名称（用于打印和保存）
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 
                          'D_B', 'G_B', 'cycle_B', 'idt_B',
                          'rec_A', 'rec_B', 'vq', 'vq_pp']  # 添加VQ相关损失
        
        # 定义要保存/显示的图像
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'recon_A']  # 添加recon_A (A->A重建)
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'recon_B']  # 添加recon_B (B->B重建)
        
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B
        
        # 定义模型名称
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']  # 只有一个生成器G
        else:
            self.model_names = ['G']

        # 定义网络
        # 使用VQ双编解码器生成器

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netGab, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        opt.n_embed, opt.embed_dim, opt.beta, opt.decay)
        
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
        
        # Identity loss
        if lambda_idt > 0:
            # G应该对真实图像保持恒等映射
            self.idt_A = self.netG(self.real_B, direction='BtoB')
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            self.idt_B = self.netG(self.real_A, direction='AtoA')
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
        
        # VQ loss
        # 获取所有前向传播中累积的VQ损失
        # 如果网络被DataParallel包装，需要通过.module访问
        vq_generator = self.netG.module if hasattr(self.netG, 'module') else self.netG
        self.loss_vq = vq_generator.get_vq_loss() * lambda_vq
        
        # VQ perplexity (用于监控，不参与反向传播)
        self.loss_vq_pp = vq_generator.perplexity.mean() if hasattr(vq_generator, 'perplexity') else 0
        
        # Combined loss
        self.loss_G = (self.loss_G_A + self.loss_G_B + 
                      self.loss_cycle_A + self.loss_cycle_B + 
                      self.loss_idt_A + self.loss_idt_B +
                      self.loss_rec_A + self.loss_rec_B +
                      self.loss_vq)
        
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
    
    def get_current_visuals(self):
        """返回当前的可视化图像"""
        visual_ret = super().get_current_visuals()
        
        # 添加VQ码本使用情况的可视化（可选）
        vq_generator = self.netG.module if hasattr(self.netG, 'module') else self.netG
        if hasattr(vq_generator, 'indices') and vq_generator.indices is not None:
            # 可以在这里添加码本使用情况的可视化
            pass
            
        return visual_ret
    
    def evaluate_codebook(self):
        """评估VQ码本的使用情况"""
        vq_generator = self.netG.module if hasattr(self.netG, 'module') else self.netG
        usage, usage_rate = vq_generator.get_codebook_usage()
        if usage is not None:
            print(f"Codebook usage rate: {usage_rate:.2%}")
            print(f"Active codes: {(usage > 0).sum().item()}/{len(usage)}")
        return usage, usage_rate

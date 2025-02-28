import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.models as models
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

class PerceptualLoss(torch.nn.Module):
    """基于VGG的感知损失，改进版"""
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # 定义要提取特征的层索引
        self.feature_layers = [1, 6, 11]  # VGG19中的conv1_2, conv2_2, conv3_2
        
        # 创建直到每个特征层的子网络
        self.slices = nn.ModuleList()
        start = 0
        for end in self.feature_layers:
            slice_model = nn.Sequential()
            for i in range(start, end + 1):
                slice_model.add_module(str(i), vgg[i])
            self.slices.append(slice_model)
            start = end + 1
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x, y):
        # 处理单通道输入 - 复制到三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
            
        # 标准化输入大小
        if x.shape[2] > 224 or x.shape[3] > 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 计算各层特征的差异
        loss = 0.0
        for slice_model in self.slices:
            x = slice_model(x)
            y = slice_model(y)
            loss += F.mse_loss(x, y)
            
        return loss

class VQCycleGANModel(BaseModel, nn.Module):
    """
    优化的CycleGAN模型，集成了Taming Transformers中的核心思想
    
    改进:
    - 增加感知损失，提高语义一致性
    - 使用自适应VQ损失权重
    - 添加判别器梯度惩罚提高稳定性
    - 使用AdamW优化器和余弦学习率调度
    - 动态调整VQ损失权重
    - 增加码本正则化
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity loss')
            parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='weight for perceptual loss')
            parser.add_argument('--lambda_vq', type=float, default=5.0, help='max weight for VQ loss')
            parser.add_argument('--vq_ramp_epochs', type=int, default=10000, help='epochs to ramp up VQ loss')
            parser.add_argument('--r1_gamma', type=float, default=10.0, help='weight for R1 gradient penalty')
            parser.add_argument('--codebook_reg', type=float, default=0.1, help='weight for codebook regularization')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        nn.Module.__init__(self)  # Initialize the nn.Module parent
        self.schedulers = []
        # 扩展loss监控
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 
                           'vq_A', 'idt_vq_A', 'vq', 'perceptual_A', 'perceptual_B', 'codebook_reg']
        
        # 指定要显示的图像
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # 如果使用identity loss
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # 合并显示
        
        # 指定要保存的模型
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # 定义网络
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netGab, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # 定义判别器
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # identity loss仅当输入和输出通道数相同时有效
                assert(opt.input_nc == opt.output_nc)
                
            # 创建图像缓冲区
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            # 创建各种loss
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.perceptual_loss = PerceptualLoss().to(self.device)
            
            # 优化器：使用AdamW替代Adam，有更好的权重衰减策略
            self.optimizer_G = torch.optim.AdamW(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-4
            )
            self.optimizer_D = torch.optim.AdamW(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-4
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            # 添加余弦退火学习率调度
            self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=opt.n_epochs, eta_min=opt.lr * 0.1)
            self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=opt.n_epochs, eta_min=opt.lr * 0.1)
            self.schedulers.append(self.scheduler_G)
            self.schedulers.append(self.scheduler_D)
            
            # VQ损失相关参数
            self.register_buffer('current_epoch', torch.tensor(0))
            self.vq_weight_max = opt.lambda_vq if hasattr(opt, 'lambda_vq') else 5.0
            self.vq_ramp_epochs = opt.vq_ramp_epochs if hasattr(opt, 'vq_ramp_epochs') else 10000
            self.r1_gamma = opt.r1_gamma if hasattr(opt, 'r1_gamma') else 10.0
            self.lambda_perceptual = opt.lambda_perceptual if hasattr(opt, 'lambda_perceptual') else 1.0
            self.codebook_reg_weight = opt.codebook_reg if hasattr(opt, 'codebook_reg') else 0.1

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """前向传播，由optimize_parameters和test调用"""
        # G_A(A)
        self.fake_B, self.loss_vq_A, self.vq_stats_A = self.netG_A(self.real_A)
        # G_B(G_A(A))
        self.rec_A = self.netG_B(self.fake_B)
        # G_B(B)
        self.fake_A = self.netG_B(self.real_B)
        # G_A(G_B(B))
        self.rec_B, _, _ = self.netG_A(self.fake_A)

    def r1_penalty(self, real_pred, real_img):
        """R1梯度惩罚，稳定GAN训练"""
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, 
            create_graph=True, retain_graph=True
        )[0]
        grad_penalty = grad_real.pow(2).reshape(real_img.size(0), -1).sum(1).mean()
        return grad_penalty

    def compute_adaptive_weight(self, recon_loss, g_loss, last_layer):
        """计算自适应权重，平衡重建与对抗损失"""
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        
        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def backward_D_basic(self, netD, real, fake):
        """计算判别器的基础损失"""
        # 确保real需要梯度计算
        real.requires_grad = True
        
        # 真实图像
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # 添加R1梯度惩罚
        r1_penalty = self.r1_penalty(pred_real, real) * self.r1_gamma
        
        # 假图像
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # 组合损失
        loss_D = (loss_D_real + loss_D_fake) * 0.5 + r1_penalty
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

    def get_current_vq_weight(self):
        """获取当前VQ损失权重，随着训练进行逐渐增加"""
        if self.current_epoch < self.vq_ramp_epochs:
            return self.vq_weight_max * (self.current_epoch / self.vq_ramp_epochs)
        else:
            return self.vq_weight_max

    def get_unwrapped_model(self, model):
        """
        Helper method to access the underlying model that might be wrapped in DataParallel.
        
        Args:
            model: A PyTorch model, potentially wrapped in DataParallel
            
        Returns:
            The unwrapped model
        """
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def backward_G(self):
        """计算生成器G_A和G_B的损失"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        # Identity loss
        if lambda_idt > 0:
            # G_A应当在输入B时保持身份映射: ||G_A(B) - B||
            self.idt_A, self.loss_idt_vq_A, _ = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            # G_B应当在输入A时保持身份映射: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            self.loss_idt_vq_A = 0

        # GAN损失D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        
        # GAN损失D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        # 计算感知损失
        self.loss_perceptual_A = self.perceptual_loss(self.rec_A, self.real_A) * self.lambda_perceptual
        self.loss_perceptual_B = self.perceptual_loss(self.rec_B, self.real_B) * self.lambda_perceptual
        
        # 前向循环一致性损失 || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A + self.loss_perceptual_A
        
        # 后向循环一致性损失 || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B + self.loss_perceptual_B
        
        # 使用自适应权重计算VQ损失
        recon_loss = self.loss_cycle_A + self.loss_cycle_B
        g_loss = self.loss_G_A + self.loss_G_B
        
        # 使用辅助方法获取模型
        unwrapped_model = self.get_unwrapped_model(self.netG_A)
        last_layer = unwrapped_model.get_last_layer()
        adaptive_weight = self.compute_adaptive_weight(recon_loss, g_loss, last_layer)
        
        # 计算VQ损失
        vq_weight = self.get_current_vq_weight() * adaptive_weight
        self.loss_vq = (self.loss_vq_A + self.loss_idt_vq_A) * vq_weight
        
        # 码本正则化
        self.loss_codebook_reg = unwrapped_model.get_codebook_reg_loss() * self.codebook_reg_weight
        
        # 组合损失并计算梯度
        self.loss_G = (self.loss_G_A + self.loss_G_B + 
                    self.loss_cycle_A + self.loss_cycle_B + 
                    self.loss_idt_A + self.loss_idt_B + 
                    self.loss_vq + self.loss_codebook_reg)
        self.loss_G.backward()

    def optimize_parameters(self):
        """计算损失，梯度，并更新网络权重"""
        # 前向传播
        self.forward()
        
        # 更新G_A和G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # 更新D_A和D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        
        # 更新epoch计数
        self.current_epoch += 1
    
    def get_current_visuals(self):
        """获取当前可视化结果与VQ统计信息"""
        visuals = super().get_current_visuals()
        
        # 添加VQ统计信息到可视化结果
        if hasattr(self, 'vq_stats_A'):
            for key, value in self.vq_stats_A.items():
                visuals[f'vq_stat_{key}'] = value.detach() if torch.is_tensor(value) else value
                
        return visuals
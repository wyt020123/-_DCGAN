import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LRScheduler
import random
import os
import logging

logger = logging.getLogger(__name__)

class WarmupCosineScheduler(LRScheduler):
    """带预热的余弦学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 
                   (1 + np.cos(np.pi * progress)) / 2 
                   for base_lr in self.base_lrs]

def weights_init_normal(m):
    """初始化网络权重"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """计算梯度惩罚"""
    # 随机权重
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # 插值样本
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # 判别器输出
    d_interpolates = discriminator(interpolates)
    
    # 计算梯度
    fake = torch.ones(real_samples.size(0), 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 计算梯度惩罚
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def spectral_norm(module, name='weight', power_iterations=1):
    """谱归一化"""
    def _make_unit_norm(W):
        W_mat = W.view(W.size(0), -1)
        with torch.no_grad():
            for _ in range(power_iterations):
                v = F.normalize(W_mat.t() @ u, dim=0)
                u = F.normalize(W_mat @ v, dim=0)
        sigma = u.t() @ W_mat @ v
        W_sn = W / sigma
        return W_sn

    u = None
    W = getattr(module, name)
    with torch.no_grad():
        u = F.normalize(torch.randn(W.size(0)), dim=0)
    
    W_sn = _make_unit_norm(W)
    setattr(module, name, W_sn)
    return module

class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化影子权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def save_checkpoint(model, optimizer, epoch, filename):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    logger.info(f"检查点已保存: {filename}")

def load_checkpoint(model, optimizer, filename):
    """加载检查点"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f"检查点已加载: {filename}")
        return epoch
    else:
        logger.error(f"检查点文件不存在: {filename}")
        return 0

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class GradientClipper:
    """梯度裁剪"""
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def __call__(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

class FeatureMatchingLoss(nn.Module):
    """特征匹配损失"""
    def __init__(self, layers=[0, 3, 6, 8, 11]):
        super(FeatureMatchingLoss, self).__init__()
        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self, features_real, features_fake):
        loss = 0
        for feat_real, feat_fake in zip(features_real, features_fake):
            loss += self.criterion(feat_real, feat_fake)
        return loss

class AdaptiveInstanceNorm(nn.Module):
    """自适应实例归一化"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习的参数
        self.weight = None
        self.bias = None
        
        # 实例归一化层
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x, style):
        # 计算风格统计量
        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True) + self.eps
        
        # 实例归一化
        x = self.instance_norm(x)
        
        # 应用风格
        return x * style_std + style_mean

def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # 生成随机裁剪框
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机选择裁剪框的中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 确保裁剪框在图像范围内
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 执行CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y = lam * y + (1 - lam) * y[index]

    return x, y, lam 
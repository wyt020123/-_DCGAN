import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# import numpy as np # 在 train.py 中未使用，可以注释掉
from model import Generator, Discriminator
from encoder import Encoder
import logging
from utils import WarmupCosineScheduler, EMA, weights_init_normal, set_seed
from visualize import TrainingVisualizer
from dataset import get_dataloader # 这条导入语句已经被使用了，不应被标记为未使用

logger = logging.getLogger(__name__)

class GANLoss(nn.Module):
    """GAN损失函数"""
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode == 'wgan':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
                
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class ReconstructionLoss(nn.Module):
    """重建损失函数"""
    def __init__(self, loss_type='l1'):
        super(ReconstructionLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'Reconstruction loss type {loss_type} not implemented')

    def __call__(self, x, y):
        return self.loss(x, y)

class PerceptualLoss(nn.Module):
    """感知损失函数"""
    def __init__(self, layers=[2, 7, 12, 21, 30]):
        super(PerceptualLoss, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.vgg_layers = vgg.features
        self.layers = layers
        self.criterion = nn.L1Loss()
        
        # 冻结VGG参数
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        
        loss = 0
        for x_feat, y_feat in zip(x_features, y_features):
            loss += self.criterion(x_feat, y_feat)
        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

class StyleLoss(nn.Module):
    """风格损失函数"""
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def forward(self, x, y):
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        
        loss = 0
        for x_feat, y_feat in zip(x_features, y_features):
            x_gram = self.gram_matrix(x_feat)
            y_gram = self.gram_matrix(y_feat)
            loss += torch.mean((x_gram - y_gram) ** 2)
        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features

class TotalLoss(nn.Module):
    """总损失函数"""
    def __init__(self, gan_mode='vanilla', recon_loss_type='l1', 
                 lambda_gan=1.0, lambda_recon=10.0, lambda_perceptual=1.0, lambda_style=1.0):
        super(TotalLoss, self).__init__()
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.recon_loss = ReconstructionLoss(loss_type=recon_loss_type)
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        
        self.lambda_gan = lambda_gan
        self.lambda_recon = lambda_recon
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style

    def forward(self, real_images, fake_images, discriminator_real, discriminator_fake, 
                reconstructed_images=None, style_images=None):
        # GAN损失
        gan_loss_real = self.gan_loss(discriminator_real, True)
        gan_loss_fake = self.gan_loss(discriminator_fake, False)
        gan_loss = (gan_loss_real + gan_loss_fake) * self.lambda_gan

        # 重建损失
        recon_loss = 0
        if reconstructed_images is not None:
            recon_loss = self.recon_loss(fake_images, reconstructed_images) * self.lambda_recon

        # 感知损失
        perceptual_loss = self.perceptual_loss(fake_images, real_images) * self.lambda_perceptual

        # 风格损失
        style_loss = 0
        if style_images is not None:
            style_loss = self.style_loss(fake_images, style_images) * self.lambda_style

        # 总损失
        total_loss = gan_loss + recon_loss + perceptual_loss + style_loss

        return {
            'total_loss': total_loss,
            'gan_loss': gan_loss,
            'recon_loss': recon_loss,
            'perceptual_loss': perceptual_loss,
            'style_loss': style_loss
        }

def train_step(generator, discriminator, encoder, optimizer_G, optimizer_D, 
               dataloader, loss_fn, device, epoch):
    """单步训练函数"""
    generator.train()
    discriminator.train()
    encoder.train()
    
    total_losses = {
        'total_loss': 0,
        'gan_loss': 0,
        'recon_loss': 0,
        'perceptual_loss': 0,
        'style_loss': 0
    }
    
    for batch_idx, (real_images, style_images) in enumerate(dataloader):
        real_images = real_images.to(device)
        style_images = style_images.to(device)
        batch_size = real_images.size(0)

        # 训练判别器
        optimizer_D.zero_grad()
        
        # 生成假图像
        z = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(z)
        
        # 判别器前向传播
        real_pred = discriminator(real_images)
        fake_pred = discriminator(fake_images.detach())
        
        # 计算判别器损失
        d_loss = loss_fn.gan_loss(real_pred, True) + loss_fn.gan_loss(fake_pred, False)
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器和编码器
        optimizer_G.zero_grad()
        
        # 重新生成假图像
        fake_images = generator(z)
        fake_pred = discriminator(fake_images)
        
        # 编码器重建
        encoded_z = encoder(real_images)
        reconstructed_images = generator(encoded_z)
        
        # 计算生成器损失
        losses = loss_fn(
            real_images=real_images,
            fake_images=fake_images,
            discriminator_real=real_pred,
            discriminator_fake=fake_pred,
            reconstructed_images=reconstructed_images,
            style_images=style_images
        )
        
        losses['total_loss'].backward()
        optimizer_G.step()

        # 更新总损失
        for k, v in losses.items():
            total_losses[k] += v.item()

        if batch_idx % 100 == 0:
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, '
                       f'D_loss: {d_loss.item():.4f}, '
                       f'G_loss: {losses["total_loss"].item():.4f}')

    # 计算平均损失
    for k in total_losses:
        total_losses[k] /= len(dataloader)

    return total_losses

def train(generator, discriminator, encoder, train_loader, 
          num_epochs=100, lr=0.0002, beta1=0.5, device='cuda'):
    """训练函数"""
    # 初始化可视化工具
    visualizer = TrainingVisualizer()
    
    # 初始化优化器
    optimizer_G = optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()),
        lr=lr, betas=(beta1, 0.999)
    )
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # 初始化学习率调度器
    scheduler_G = WarmupCosineScheduler(optimizer_G, warmup_epochs=5, max_epochs=num_epochs)
    scheduler_D = WarmupCosineScheduler(optimizer_D, warmup_epochs=5, max_epochs=num_epochs)

    # 初始化损失函数
    loss_fn = TotalLoss(
        gan_mode='vanilla',
        recon_loss_type='l1',
        lambda_gan=1.0,
        lambda_recon=10.0,
        lambda_perceptual=1.0,
        lambda_style=1.0
    )

    # 初始化EMA
    ema = EMA(generator, decay=0.999)

    # 训练循环
    for epoch in range(num_epochs):
        losses = train_step(
            generator=generator,
            discriminator=discriminator,
            encoder=encoder,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            dataloader=train_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch
        )

        # 更新学习率
        scheduler_G.step()
        scheduler_D.step()
        
        # 更新EMA
        ema.update()

        # 更新可视化工具
        visualizer.update_losses(losses)
        
        # 绘制损失曲线
        visualizer.plot_losses(epoch)
        visualizer.plot_separate_losses(epoch)
        
        # 生成一些示例图像
        with torch.no_grad():
            z = torch.randn(16, 100, 1, 1, device=device)
            fake_images = generator(z)
            visualizer.save_generated_images(fake_images, epoch)
        
        # 生成训练报告
        model_info = {
            'Generator Parameters': sum(p.numel() for p in generator.parameters()),
            'Discriminator Parameters': sum(p.numel() for p in discriminator.parameters()),
            'Encoder Parameters': sum(p.numel() for p in encoder.parameters()),
            'Learning Rate': optimizer_G.param_groups[0]['lr'],
            'Batch Size': train_loader.batch_size
        }
        visualizer.generate_training_report(epoch, model_info)

        # 保存模型检查点
        if (epoch + 1) % 10 == 0:
            # 使用EMA模型进行保存
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'losses': losses
            }, f'checkpoints/model_epoch_{epoch+1}.pth')
            ema.restore()

    # 绘制学习率调度曲线
    visualizer.plot_lr_schedule(optimizer_G, num_epochs)

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    set_seed(42)
    
    # 初始化模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    encoder = Encoder(latent_dim=100).to(device)
    
    # 初始化权重
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    encoder.apply(weights_init_normal)
    
    # 创建数据加载器
    train_loader = get_dataloader(
        image_folder='./images', # 请将此路径替换为你实际的数据集路径
        batch_size=32,
        image_size=256 # 根据你的模型输入调整图像大小
    )
    
    # 开始训练
    train(
        generator=generator,
        discriminator=discriminator,
        encoder=encoder,
        train_loader=train_loader,
        num_epochs=100,
        lr=0.0002,
        beta1=0.5,
        device=device
    ) 
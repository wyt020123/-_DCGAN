import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import lpips

class MetricsCalculator:
    """计算各种评估指标"""
    def __init__(self, device='cuda'):
        self.device = device
        # 初始化LPIPS模型用于计算感知相似度
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
    def calculate_psnr(self, img1, img2):
        """计算PSNR (Peak Signal-to-Noise Ratio)"""
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    def calculate_ssim(self, img1, img2, window_size=11, size_average=True):
        """计算SSIM (Structural Similarity Index)"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 创建高斯窗口
        window = torch.Tensor([np.exp(-(x - window_size//2)**2/float(window_size)) for x in range(window_size)])
        window = window/window.sum()
        window = window.unsqueeze(1)
        window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
        window = window.expand(img1.size(1), 1, window_size, window_size).to(self.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean().item()
        return ssim_map.mean(1).mean(1).mean(1).item()
    
    def calculate_lpips(self, img1, img2):
        """计算LPIPS (Learned Perceptual Image Patch Similarity)"""
        return self.lpips_model(img1, img2).item()
    
    def calculate_fid(self, real_features, fake_features):
        """计算FID (Fréchet Inception Distance)"""
        mu_real = torch.mean(real_features, dim=0)
        mu_fake = torch.mean(fake_features, dim=0)
        sigma_real = torch.cov(real_features.t())
        sigma_fake = torch.cov(fake_features.t())
        
        ssdiff = torch.sum((mu_real - mu_fake) ** 2.0)
        covmean = torch.matrix_power(sigma_real @ sigma_fake, 0.5)
        
        if torch.isnan(covmean).any():
            covmean = torch.zeros_like(covmean)
            
        fid = ssdiff + torch.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return fid.item()
    
    def calculate_inception_score(self, images, model, splits=10):
        """计算Inception Score"""
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(images), 100):
                batch = images[i:i+100]
                pred = F.softmax(model(batch), dim=1)
                preds.append(pred)
        preds = torch.cat(preds, 0)
        
        # 计算每个分割的分数
        scores = []
        for i in range(splits):
            part = preds[i * (len(preds) // splits):(i + 1) * (len(preds) // splits)]
            kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0, keepdim=True)))
            kl = torch.mean(torch.sum(kl, dim=1))
            scores.append(torch.exp(kl).item())
        
        return np.mean(scores), np.std(scores)

class LossCalculator:
    """计算各种损失函数"""
    def __init__(self, device='cuda'):
        self.device = device
        # 初始化VGG模型用于计算感知损失
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.vgg_layers = vgg[:20]  # 使用前20层
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
    def content_loss(self, real_img, fake_img):
        """计算内容损失（使用VGG特征）"""
        real_features = self.get_features(real_img)
        fake_features = self.get_features(fake_img)
        return F.mse_loss(real_features, fake_features)
    
    def style_loss(self, real_img, fake_img):
        """计算风格损失（使用Gram矩阵）"""
        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram.div(c * h * w)
            
        real_features = self.get_features(real_img)
        fake_features = self.get_features(fake_img)
        
        real_gram = gram_matrix(real_features)
        fake_gram = gram_matrix(fake_features)
        
        return F.mse_loss(real_gram, fake_gram)
    
    def adversarial_loss(self, real_pred, fake_pred, target_real=1.0, target_fake=0.0):
        """计算对抗损失"""
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred) * target_real)
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred) * target_fake)
        return real_loss + fake_loss
    
    def reconstruction_loss(self, real_img, fake_img, loss_type='l1'):
        """计算重建损失"""
        if loss_type == 'l1':
            return F.l1_loss(real_img, fake_img)
        elif loss_type == 'l2':
            return F.mse_loss(real_img, fake_img)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def perceptual_loss(self, real_img, fake_img):
        """计算感知损失（使用LPIPS）"""
        lpips_model = lpips.LPIPS(net='alex').to(self.device)
        return lpips_model(real_img, fake_img).mean()
    
    def get_features(self, x):
        """提取VGG特征"""
        features = []
        for layer in self.vgg_layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features[-1]  # 返回最后一层特征

def calculate_accuracy(predictions, targets):
    """计算分类准确率"""
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def calculate_precision_recall(predictions, targets, threshold=0.5):
    """计算精确率和召回率"""
    predictions = (predictions > threshold).float()
    true_positives = (predictions * targets).sum().item()
    false_positives = (predictions * (1 - targets)).sum().item()
    false_negatives = ((1 - predictions) * targets).sum().item()
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    
    return precision, recall

def calculate_f1_score(precision, recall):
    """计算F1分数"""
    return 2 * (precision * recall) / (precision + recall + 1e-8)

if __name__ == '__main__':
    # 使用示例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化计算器
    metrics_calculator = MetricsCalculator(device)
    loss_calculator = LossCalculator(device)
    
    # 创建示例数据
    batch_size = 4
    real_images = torch.randn(batch_size, 3, 256, 256).to(device)
    fake_images = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # 计算各种指标
    psnr = metrics_calculator.calculate_psnr(real_images, fake_images)
    ssim = metrics_calculator.calculate_ssim(real_images, fake_images)
    lpips_score = metrics_calculator.calculate_lpips(real_images, fake_images)
    
    # 计算各种损失
    content_loss = loss_calculator.content_loss(real_images, fake_images)
    style_loss = loss_calculator.style_loss(real_images, fake_images)
    perceptual_loss = loss_calculator.perceptual_loss(real_images, fake_images)
    reconstruction_loss = loss_calculator.reconstruction_loss(real_images, fake_images)
    
    print(f"PSNR: {psnr:.2f}")
    print(f"SSIM: {ssim:.4f}")
    print(f"LPIPS: {lpips_score:.4f}")
    print(f"Content Loss: {content_loss:.4f}")
    print(f"Style Loss: {style_loss:.4f}")
    print(f"Perceptual Loss: {perceptual_loss:.4f}")
    print(f"Reconstruction Loss: {reconstruction_loss:.4f}") 
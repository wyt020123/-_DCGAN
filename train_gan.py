import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 超参数设置
image_size = 256
batch_size = 128
latent_dim = 100
num_epochs = 20
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径（你的Anime Faces数据集路径）
data_path = r"images"  # 注意：ImageFolder要求下一级是类别名文件夹

# 输出目录
os.makedirs("generated", exist_ok=True)
os.makedirs("training_plots", exist_ok=True)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # 归一化到 [-1, 1] 之间
])

# 加载数据
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 显示图像的辅助函数
def imshow(img):
    img = img * 0.5 + 0.5  # 反归一化到 [0, 1] 区间
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # 不显示坐标轴
    plt.show()


# 测试图片加载是否成功
try:
    real_batch = next(iter(dataloader))
    print(f"数据加载成功，数据集大小: {len(dataset)}")
    imshow(torchvision.utils.make_grid(real_batch[0][:16]))
except Exception as e:
    print(f"数据加载失败: {e}")
    print("请检查数据路径是否正确，以及数据集格式是否符合要求。")


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: [batch_size, latent_dim, 1, 1]
            nn.ConvTranspose2d(latent_dim, 2048, 4, 1, 0, bias=False),  # -> [2048, 4, 4]
            nn.BatchNorm2d(2048),
            nn.ReLU(True),

            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),  # -> [1024, 8, 8]
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # -> [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # -> [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # -> [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # -> [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # -> [3, 256, 256]
            nn.Tanh()  # Tanh 激活，将图像输出映射到 [-1, 1] 区间
        )

    def forward(self, input):
        # 打印输入和输出形状以便调试
        print("Generator input shape:", input.shape)
        output = self.main(input)
        print("Generator output shape:", output.shape)
        return output


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: [3, 256, 256]
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # -> [64, 128, 128]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # -> [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # -> [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # -> [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # -> [1024, 8, 8]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),  # -> [2048, 4, 4]
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2048, 1, 4, 1, 0, bias=False),  # -> [1, 1, 1]
            nn.Sigmoid()  # Sigmoid 输出概率
        )

    def forward(self, input):
        # 打印输入和输出形状以便调试
        print("Discriminator input shape:", input.shape)
        output = self.main(input)
        print("Discriminator output shape before view:", output.shape)
        output = output.view(-1, 1)  # 保持维度为 [batch_size, 1]
        print("Discriminator output shape after view:", output.shape)
        return output


# 模型初始化
netG = Generator().to(device)
netD = Discriminator().to(device)

# 损失函数与优化器
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 固定噪声用于生成并显示固定图像
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)


def plot_training_statistics(G_losses, D_losses, D_accuracies, D_fake_accuracies):
    """绘制训练统计数据图表"""
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(G_losses, label='生成器损失')
    plt.plot(D_losses, label='判别器损失')
    plt.title('训练损失')
    plt.xlabel('轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(D_accuracies, label='判别器总准确率', color='green')
    # 计算生成器欺骗判别器的准确率 (1 - 判别器对假样本的准确率)
    G_fooling_accuracies = [1 - acc for acc in D_fake_accuracies]
    plt.plot(G_fooling_accuracies, label='生成器欺骗准确率', color='purple')
    plt.axhline(y=0.5, color='r', linestyle='--', label='随机猜测')
    plt.title('判别器与生成器准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = 'training_plots/training_statistics.png'
    print(f"尝试保存训练统计图表至: {save_path}")
    try:
        plt.savefig(save_path)
        print("已成功保存训练统计图表")
    except Exception as e:
        print(f"保存图表时出错: {e}")
    plt.close()


# 开始训练
print("开始训练...")

# 用于存储训练过程中的统计数据
G_losses = []
D_losses = []
D_total_accuracies = []
D_fake_accuracies = []

for epoch in range(num_epochs):
    epoch_G_loss = 0
    epoch_D_loss = 0
    epoch_D_total_acc = 0
    epoch_D_fake_acc = 0
    batches_done = 0

    for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        real_labels = torch.ones(b_size, 1, device=device)
        fake_labels = torch.zeros(b_size, 1, device=device)

        # 训练判别器
        netD.zero_grad()
        output_real = netD(real_images)
        loss_real = criterion(output_real, real_labels)

        # 计算判别器对真实图像的准确率
        pred_real = output_real > 0.5
        acc_real = pred_real.float().mean().item()

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())  # 生成的图片不参与反向传播
        loss_fake = criterion(output_fake, fake_labels)

        # 计算判别器对生成图像的准确率
        pred_fake = output_fake <= 0.5
        acc_fake = pred_fake.float().mean().item()

        # 计算判别器的总损失和准确率
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # 计算判别器的总准确率
        acc_total = (acc_real + acc_fake) / 2

        # 训练生成器
        netG.zero_grad()
        output = netD(fake_images)
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        optimizerG.step()

        # 记录本轮的损失和准确率
        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()
        epoch_D_total_acc += acc_total
        epoch_D_fake_acc += acc_fake
        batches_done += 1

        # 每50步输出一次
        if i % 50 == 0:
            print(f"[Epoch {epoch + 1}/{num_epochs}] Step {i}/{len(dataloader)} "
                  f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} "
                  f"D_Acc: {acc_total:.4f}")

        # 保存生成图像
        if i % 100 == 0:  # 每100步保存一张生成图像
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                save_image(fake, f"generated/fake_images_epoch{epoch + 1}_step{i}.png", normalize=True)

    # 计算平均损失和准确率
    avg_G_loss = epoch_G_loss / batches_done
    avg_D_loss = epoch_D_loss / batches_done
    avg_D_total_acc = epoch_D_total_acc / batches_done
    avg_D_fake_acc = epoch_D_fake_acc / batches_done

    # 记录统计数据
    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)
    D_total_accuracies.append(avg_D_total_acc)
    D_fake_accuracies.append(avg_D_fake_acc)

    print(f'Epoch [{epoch + 1}/{num_epochs}] 生成器平均损失: {avg_G_loss:.4f}, 判别器平均损失: {avg_D_loss:.4f}, '
          f'判别器平均准确率: {avg_D_total_acc:.4f}')

    # 每一轮结束时保存生成的图像
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        save_image(fake, f"generated/fake_images_epoch_{epoch + 1}.png", normalize=True)

# 训练结束后绘制损失曲线和准确率曲线
plot_training_statistics(G_losses, D_losses, D_total_accuracies, D_fake_accuracies)

# 保存训练好的模型
torch.save(netG.state_dict(), "generator.pth")
torch.save(netD.state_dict(), "discriminator.pth")
print("训练完成，模型已保存。")
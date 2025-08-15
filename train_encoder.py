# train_encoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder, Generator  # 你已经写好的网络
from dataset import get_dataloader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
image_folder = "D:/GAN_Project/images"   # 注意：ImageFolder 要求有子文件夹
batch_size = 64
latent_dim = 100
num_epochs = 50
lr = 0.0002

# 加载数据
dataloader = get_dataloader(image_folder, batch_size=batch_size)

# 加载网络
encoder = Encoder(latent_dim=latent_dim).to(device)
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))  # 加载预训练的生成器
generator.eval()  # 不训练 Generator

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters(), lr=lr)

# 训练循环
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        # 生成 latent 向量并还原图像
        z = encoder(imgs)
        z = z.view(z.size(0), latent_dim, 1, 1)
        recon_imgs = generator(z)

        loss = criterion(recon_imgs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # 每轮保存一次模型
    torch.save(encoder.state_dict(), "encoder.pth")
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        # 打印输入形状
        print("Input shape:", x.shape)
        
        # 通过encoder
        x = self.encoder(x)  # (B, 512, 4, 4)
        print("After encoder shape:", x.shape)
        
        # 展平
        x = x.view(x.size(0), -1)  # (B, 512*4*4)
        
        # 通过全连接层
        latent = self.fc(x)  # (B, latent_dim)
        
        # 调整形状为生成器需要的格式
        latent = latent.view(latent.size(0), self.latent_dim, 1, 1)  # (B, latent_dim, 1, 1)
        print("Output shape:", latent.shape)

        return latent
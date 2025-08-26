import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D CNN分類モデル
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 14 * 10 * 64, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d12 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(32)
        self.batchnorm3d3 = nn.BatchNorm3d(64)
        self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = F.relu(self.batchnorm3d12(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d2(self.conv3(x)))
        x = F.relu(self.batchnorm3d3(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 14 * 10 * 64)
        x = self.dropout(x)
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# CNNベースAutoEncoder定義
class CNNAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14x14 → 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 → 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)   # 潜在表現
        x_recon = self.decoder(z)  # 再構成画像
        return x_recon, z

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
class CNNAutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),  # 80->40,112->56,80->40
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), # 40->20,56->28,40->20
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), # 20->10,28->14,20->10
            nn.ReLU(True)
        )

        # Flatten
        self.flatten_dim = 64*10*14*10
        self.fc1 = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=(0,0,0)),  # 10->20,14->28,10->20
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=(0,0,0)),  # 20->40,28->56,20->40
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=(0,0,0)),   # 40->80,56->112,40->80
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decode(self, z):
        x = self.fc2(z)
        x = x.view(x.size(0), 64, 10, 14, 10)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)



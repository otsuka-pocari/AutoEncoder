import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),  # (B,16,D/2,H/2,W/2)
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), # (B,32,D/4,H/4,W/4)
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), # (B,64,D/8,H/8,W/8)
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(64*10*14*10, latent_dim)   # flatten後にlatent_dimへ
        self.fc_decode = nn.Linear(latent_dim, 64*10*14*10)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        z = self.fc_mu(x)
        x = self.fc_decode(z)
        x = x.view(batch_size, 64, 10, 14, 10)  # reshape back
        x = self.decoder(x)
        return x

class DeepCNNAutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder (deeper)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),   # (B,16,D/2,H/2,W/2)
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),  # (B,32,D/4,H/4,W/4)
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),  # (B,64,D/8,H/8,W/8)
            nn.ReLU(True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), # (B,128,D/16,H/16,W/16)
            nn.ReLU(True),
        )
        # flatten size after encoder
        self.fc_mu = nn.Linear(128*5*7*5, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128*5*7*5)

        # Decoder (symmetric to encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        z = self.fc_mu(x)
        x = self.fc_decode(z)
        x = x.view(batch_size, 128, 5, 7, 5)  # reshape before decoding
        x = self.decoder(x)
        return x



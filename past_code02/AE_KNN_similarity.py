import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import nibabel as nib
import glob
import os

# ===========================
# 1. データセット定義
# ===========================
class ADNIDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.nii*")))
        self.labels = [0 if "CN" in f else 1 for f in self.files]  # CN=0, AD=1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = nib.load(self.files[idx]).get_fdata()
        img = np.nan_to_num(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # shape: (1, D, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# ===========================
# 2. AutoEncoder 定義
# ===========================
class CNNAutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # ----- Encoder -----
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)

        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(64)

        # エンコード後の空間サイズを推定（入力が 80×112×80）
        self.flatten_dim = 10 * 14 * 10 * 64

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=(1, 0, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=(1, 0, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(True),

            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=(1, 0, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # ----- Encoder -----
        x = self.pool(x)                                 # 80x112x80 → 40x56x40
        x = F.relu(self.batchnorm3d1(self.conv1(x)))     # 1→3
        x = F.relu(self.batchnorm3d2(self.conv2(x)))     # 3→3
        x = self.pool(x)                                 # 20x28x20
        x = F.relu(self.batchnorm3d3(self.conv3(x)))     # 3→32
        x = F.relu(self.batchnorm3d4(self.conv4(x)))     # 32→64
        x = self.pool(x)                                 # 10x14x10

        x = x.view(batch_size, -1)
        z = self.fc_mu(x)

        # ----- Decoder -----
        x = self.fc_decode(z)
        x = x.view(batch_size, 64, 10, 14, 10)
        x = self.decoder(x)

        # 最後に (80,112,80) に補正
        x = F.interpolate(x, size=(80, 112, 80), mode='trilinear', align_corners=False)

        return x, z


# ===========================
# 3. 学習と評価
# ===========================
def train_autoencoder(model, dataloader, device, epochs=20, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            x_recon, _ = model(x_batch)
            loss = criterion(x_recon, x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(dataloader):.6f}")

    print("✅ AutoEncoder 学習完了")
    return model


def extract_latent_vectors(model, dataloader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            _, z = model(x_batch)
            features.append(z.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.concatenate(features), np.concatenate(labels)


def knn_classify(train_z, train_y, test_z, test_y, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_z, train_y)
    pred = knn.predict(test_z)
    acc = accuracy_score(test_y, pred)
    print(f"✅ KNN 精度: {acc*100:.2f}%")
    return acc


# ===========================
# 4. メイン処理
# ===========================
if __name__ == "__main__":
    data_dir_train = "/path/to/ADNI/train"  # ← 学習用フォルダ
    data_dir_test = "/path/to/ADNI/test"    # ← テスト用フォルダ

    train_dataset = ADNIDataset(data_dir_train)
    test_dataset = ADNIDataset(data_dir_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNAutoEncoder3D(latent_dim=32).to(device)

    # ----- 学習 -----
    model = train_autoencoder(model, train_loader, device, epochs=20, lr=1e-4)
    torch.save(model.state_dict(), "best_autoencoder.pth")

    # ----- 潜在変数抽出 -----
    train_z, train_y = extract_latent_vectors(model, train_loader, device)
    test_z, test_y = extract_latent_vectors(model, test_loader, device)

    # ----- KNN 分類 -----
    knn_classify(train_z, train_y, test_z, test_y)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

# ADNI2データ読み込み用
from datasets.load_adni import load_adni2


# =====================================
# 1️⃣ AutoEncoder定義（既存と同じ）
# =====================================
class CNNAutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=16):  # PCAを使うのでlatent_dimを小さくしてOK
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(64 * 10 * 14 * 10, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 10 * 14 * 10)
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
        x = x.view(batch_size, 64, 10, 14, 10)
        x = self.decoder(x)
        return x, z


# =====================================
# 2️⃣ ADNI2データ読み込み
# =====================================
CLASS_MAP = {"CN": 0, "AD": 1}

print("Loading ADNI2 dataset ...")
dataset_adni = load_adni2(classes=["CN", "AD"], size="half", unique=True, mni=False, strength=["3.0"])
print("Loaded dataset size:", len(dataset_adni))

voxels = np.zeros((len(dataset_adni), 80, 112, 80))
labels = np.zeros(len(dataset_adni))
for i in tqdm(range(len(dataset_adni))):
    voxels[i] = dataset_adni[i]["voxel"]
    labels[i] = CLASS_MAP[dataset_adni[i]["class"]]

X_flat = voxels.reshape(len(voxels), -1)  # flatten for PCA

# =====================================
# 3️⃣ PCAによる線形部分の抽出
# =====================================
pca_dim = 50  # PCAの次元数（調整可能）
print(f"Running PCA (dim={pca_dim}) ...")
pca = PCA(n_components=pca_dim)
X_pca = pca.fit_transform(X_flat)
X_recon_pca = pca.inverse_transform(X_pca)

# 残差 = 入力 - PCA復元
X_residual = X_flat - X_recon_pca
X_residual = X_residual.reshape(len(X_residual), 1, 80, 112, 80)

print("Residual data shape:", X_residual.shape)

# torch.Tensorへ変換
X_tensor = torch.tensor(X_residual, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# train/val分割（80/20）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False)


# =====================================
# 4️⃣ AutoEncoderを残差データで学習
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNAutoEncoder3D(latent_dim=16).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
os.makedirs("models_residual", exist_ok=True)

for epoch in range(10):
    model.train()
    train_loss = 0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        x_recon, _ = model(x_batch)
        loss = criterion(x_recon, x_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            x_recon, _ = model(x_batch)
            loss = criterion(x_recon, x_batch)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models_residual/best_model.pth")
        print("✅ モデルを保存しました。")


# =====================================
# 5️⃣ 潜在表現抽出 & KNNで評価
# =====================================
model.load_state_dict(torch.load("models_residual/best_model.pth"))
model.eval()

with torch.no_grad():
    all_z = []
    all_y = []
    for x_batch, y_batch in DataLoader(dataset, batch_size=4):
        x_batch = x_batch.to(device)
        _, z = model(x_batch)
        all_z.append(z.cpu())
        all_y.append(y_batch)
    all_z = torch.cat(all_z).numpy()
    all_y = torch.cat(all_y).numpy()

# KNN評価
knn = KNeighborsClassifier(n_neighbors=5)
train_z, test_z = all_z[:int(0.8 * len(all_z))], all_z[int(0.8 * len(all_z)):]
train_y, test_y = all_y[:int(0.8 * len(all_y))], all_y[int(0.8 * len(all_y)):]

knn.fit(train_z, train_y)
pred_y = knn.predict(test_z)
acc = accuracy_score(test_y, pred_y)
print(f"\n🔍 KNNによる潜在残差空間での一致率: {acc*100:.2f}%")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------------
# 1️⃣ AutoEncoder 定義
# -----------------------------
class CNNAutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(64 * 10 * 10 * 10, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 10 * 10 * 10)

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
        x = x.view(batch_size, 64, 10, 10, 10)
        x = self.decoder(x)
        return x, z  # zを返すようにする（潜在表現）


# -----------------------------
# 2️⃣ データ作成（仮データ）
# -----------------------------
# 例: 100サンプル, 各サンプルは(1,80,80,80)
# クラスは 0=健常, 1=AD
X = torch.randn(100, 1, 80, 80, 80)
y = torch.cat([torch.zeros(50), torch.ones(50)])  # 50:50クラス分布

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False)


# -----------------------------
# 3️⃣ 学習とモデル保存
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNAutoEncoder3D(latent_dim=32).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
os.makedirs("models", exist_ok=True)

for epoch in range(10):  # 学習10エポック（例）
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

    # 最良モデルを保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ モデルを保存しました。")


# -----------------------------
# 4️⃣ 潜在表現抽出
# -----------------------------
model.load_state_dict(torch.load("models/best_model.pth"))
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

# -----------------------------
# 5️⃣ KNNで類似症例探索
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
train_z, test_z = all_z[:80], all_z[80:]
train_y, test_y = all_y[:80], all_y[80:]

knn.fit(train_z, train_y)
pred_y = knn.predict(test_z)
acc = accuracy_score(test_y, pred_y)
print(f"\n🔍 KNNによる類似症例一致率: {acc*100:.2f}%")

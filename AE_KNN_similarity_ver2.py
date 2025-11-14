import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets.load_adni import load_adni2  # ★ 実データ読み込み用
import os

# =====================================================
# GPU設定
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# データ読み込み（ADNI2データ）
# =====================================================
dataset_adni = load_adni2(
    classes=["CN", "AD"],
    size="half",
    unique=True,
    mni=False,
    strength=["3.0"]
)


X = []
y = []

# 各被験者からボクセルとラベルを抽出
for subject in dataset_adni:
    voxel = subject["voxel"]
    label = 0 if subject["class"] == "CN" else 1  # CN=0, AD=1
    X.append(voxel)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"Loaded ADNI2 dataset: X={X.shape}, y={y.shape}")

# =====================================================
# 前処理
# =====================================================
X = (X - X.min()) / (X.max() - X.min())  # 正規化
X = np.expand_dims(X, axis=1)  # (N, 1, 80, 112, 80)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# =====================================================
# AutoEncoder モデル定義（3D CNN）
# =====================================================
class AutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super(AutoEncoder3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),  # (80,112,80)→(40,56,40)
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), # →(20,28,20)
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), # →(10,14,10)
            nn.ReLU()
        )
        self.flatten_dim = 64 * 10 * 14 * 10  # encoder出力サイズ
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1), # →(20,28,20)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1), # →(40,56,40)
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1),  # →(80,112,80)
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

# =====================================================
# 学習設定
# =====================================================
latent_dim = 32
model = AutoEncoder3D(latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =====================================================
# 学習ループ
# =====================================================
num_epochs = 30
save_path = "best_model.pth"
best_loss = float("inf")

print("Training start ...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)
        x_recon, _ = model(x_batch)
        loss = criterion(x_recon, x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    # ベストモデル保存
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print("  ✅ Model saved (best so far)")

print("Training finished!")

# =====================================================
# 特徴抽出
# =====================================================
model.load_state_dict(torch.load(save_path))
model.eval()

def extract_features(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            _, z = model(x_batch)
            feats.append(z.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.concatenate(feats), np.concatenate(labels)

train_feats, train_labels = extract_features(train_loader)
test_feats, test_labels = extract_features(test_loader)
print("Feature extraction done:", train_feats.shape, test_feats.shape)

# =====================================================
# KNN分類
# =====================================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_feats, train_labels)
pred = knn.predict(test_feats)

acc = accuracy_score(test_labels, pred)
print(f"\nKNN accuracy: {acc:.4f}")

# =====================================================
# 結果出力
# =====================================================
print("All done. Model and features ready.")


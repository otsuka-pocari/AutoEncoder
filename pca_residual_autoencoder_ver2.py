import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datasets.load_adni import load_adni2

# ------------------------
# 設定
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# PCA / AE の次元設定（必要に応じて調整）
pca_dim = 128            # PCA で残す線形成分（例: 128）
res_pca_dim = 64         # 残差に対して行う第二の PCA 次元（これが AE の入力次元）
ae_latent = 32           # Residual AE の潜在次元（最終特徴として使用）
ae_batch_size = 32       # AE のバッチサイズ（メモリに応じて下げてください）
net_batch_size = 2       # LuckyNet のバッチサイズ（3D CNN は重いので小さめ）

# ------------------------
# データ読み込み（ADNI2）
# ------------------------
dataset_adni = load_adni2(
    classes=["CN", "AD"],
    size="half",
    unique=True,
    mni=False,
    strength=["3.0"]
)

X_list = []
y_list = []
for s in dataset_adni:
    X_list.append(s["voxel"])
    y_list.append(0 if s["class"] == "CN" else 1)

X = np.array(X_list, dtype=np.float32)  # (N, D1, D2, D3)
y = np.array(y_list, dtype=np.int64)

N, D1, D2, D3 = X.shape
voxel_dim = D1 * D2 * D3
print("Loaded: X.shape=", X.shape, "y.shape=", y.shape)
print("voxel_dim:", voxel_dim)

# ------------------------
# 前処理：正規化
# ------------------------
X = (X - X.min()) / (X.max() - X.min())  # [0,1] 正規化

# ------------------------
# 1回だけ train/test 分割（データ漏洩回避）
# ------------------------
idx = np.arange(N)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

X_train = X[train_idx]      # (n_train, D1, D2, D3)
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

n_train = X_train.shape[0]
n_test  = X_test.shape[0]
print("Train / Test:", n_train, n_test)

# ------------------------
# PCA1: training フラットデータで fit（線形成分抽出）
# ------------------------
# sklearn PCA を使うため flatten する
X_train_flat = X_train.reshape(n_train, voxel_dim)
X_test_flat  = X_test.reshape(n_test, voxel_dim)

print("Fitting PCA1 on training set (this may take time)...")
pca1 = PCA(n_components=pca_dim, svd_solver='randomized', random_state=0)
X_train_pca = pca1.fit_transform(X_train_flat)   # (n_train, pca_dim)
X_test_pca  = pca1.transform(X_test_flat)        # (n_test, pca_dim)

# PCA1 再構成（線形復元）
X_train_pca_recon = pca1.inverse_transform(X_train_pca)  # (n_train, voxel_dim)
X_test_pca_recon  = pca1.inverse_transform(X_test_pca)   # (n_test, voxel_dim)

# ------------------------
# 残差（train/test）
# ------------------------
resid_train = X_train_flat - X_train_pca_recon  # (n_train, voxel_dim)
resid_test  = X_test_flat  - X_test_pca_recon   # (n_test, voxel_dim)

# cast float32
resid_train = resid_train.astype(np.float32)
resid_test  = resid_test.astype(np.float32)

print("Residual shapes:", resid_train.shape, resid_test.shape)

# ------------------------
# Residual PCA（train に fit）→ AE に入れる次元を小さくする
# ------------------------
print("Fitting PCA on residuals (train) ...")
res_pca = PCA(n_components=res_pca_dim, svd_solver='randomized', random_state=0)
res_train_coeff = res_pca.fit_transform(resid_train)  # (n_train, res_pca_dim)
res_test_coeff  = res_pca.transform(resid_test)       # (n_test, res_pca_dim)

print("Residual PCA coeff shapes:", res_train_coeff.shape, res_test_coeff.shape)

# ------------------------
# Residual AE (全結合) - 入力は res_pca_dim に縮小された係数
# ------------------------
class ResidualAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

ae_model = ResidualAE(input_dim=res_pca_dim, latent_dim=ae_latent).to(device)
ae_opt = optim.Adam(ae_model.parameters(), lr=1e-3)
ae_loss_fn = nn.MSELoss()

# AE 用 DataLoader（バッチサイズは ae_batch_size）
ae_train_loader = DataLoader(
    TensorDataset(torch.tensor(res_train_coeff, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
    batch_size=ae_batch_size, shuffle=True, drop_last=False
)
ae_test_loader = DataLoader(
    TensorDataset(torch.tensor(res_test_coeff, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.long)),
    batch_size=ae_batch_size, shuffle=False, drop_last=False
)

# ------------------------
# Train Residual AE
# ------------------------
num_epochs_ae = 20
print("\nTraining Residual AE ...")
for ep in range(num_epochs_ae):
    ae_model.train()
    total_loss = 0.0
    for xb, _ in ae_train_loader:
        xb = xb.to(device)
        xr, _ = ae_model(xb)
        loss = ae_loss_fn(xr, xb)

        ae_opt.zero_grad()
        loss.backward()
        ae_opt.step()
        total_loss += loss.item()
    avg = total_loss / len(ae_train_loader)
    print(f"AE Epoch {ep+1}/{num_epochs_ae} Loss={avg:.6f}")

# AE の特徴（潜在ベクトル）抽出
ae_model.eval()
with torch.no_grad():
    ae_train_feats = ae_model.encoder(torch.tensor(res_train_coeff, dtype=torch.float32).to(device)).cpu().numpy()  # (n_train, ae_latent)
    ae_test_feats  = ae_model.encoder(torch.tensor(res_test_coeff, dtype=torch.float32).to(device)).cpu().numpy()   # (n_test, ae_latent)

print("AE latent shapes:", ae_train_feats.shape, ae_test_feats.shape)

# ------------------------
# LuckyNet (3D CNN) - あなたの元の構造（出力は (logits, feat512)）
# ------------------------
class LuckyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, padding=1)
        # after 3 pools: (80,112,80) -> (40,56,40) -> (20,28,20) -> (10,14,10)
        self.flatten_dim = 32 * 10 * 14 * 10
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.25)
        self.bn3d1 = nn.BatchNorm3d(3)
        self.bn3d2 = nn.BatchNorm3d(3)
        self.bn3d3 = nn.BatchNorm3d(32)
        self.bn3d4 = nn.BatchNorm3d(32)
        self.bn1d = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.bn3d1(self.conv1(x)))
        x = F.relu(self.bn3d2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3d3(self.conv3(x)))
        x = F.relu(self.bn3d4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, self.flatten_dim)
        x = self.dropout(x)
        feat = F.relu(self.bn1d(self.fc1(x)))  # 512-dim feature
        x = self.dropout(feat)
        out = self.fc2(x)
        return out, feat

# LuckyNet の DataLoader（注意：入れるテンソルは (N,1,D1,D2,D3)）
net_train_loader = DataLoader(
    TensorDataset(torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
    batch_size=net_batch_size, shuffle=True
)
net_test_loader = DataLoader(
    TensorDataset(torch.tensor(np.expand_dims(X_test, axis=1), dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.long)),
    batch_size=net_batch_size, shuffle=False
)

# ------------------------
# LuckyNet を短く学習（必要ならエポック増やしてください）
# ------------------------
net_model = LuckyNet().to(device)
net_opt = optim.Adam(net_model.parameters(), lr=1e-4)
net_loss_fn = nn.CrossEntropyLoss()

num_epochs_net = 10
print("\nTraining LuckyNet ...")
for ep in range(num_epochs_net):
    net_model.train()
    total_loss = 0.0
    for xb, yb in net_train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, _ = net_model(xb)
        loss = net_loss_fn(logits, yb)
        net_opt.zero_grad()
        loss.backward()
        net_opt.step()
        total_loss += loss.item()
    avg = total_loss / len(net_train_loader)
    print(f"LuckyNet Epoch {ep+1}/{num_epochs_net} Loss={avg:.6f}")

# LuckyNet の特徴抽出（512次元）
net_model.eval()
with torch.no_grad():
    net_train_feats_list = []
    for xb, _ in net_train_loader:
        xb = xb.to(device)
        _, f = net_model(xb)
        net_train_feats_list.append(f.cpu().numpy())
    net_train_feats = np.concatenate(net_train_feats_list, axis=0)  # (n_train, 512)

    net_test_feats_list = []
    for xb, _ in net_test_loader:
        xb = xb.to(device)
        _, f = net_model(xb)
        net_test_feats_list.append(f.cpu().numpy())
    net_test_feats = np.concatenate(net_test_feats_list, axis=0)    # (n_test, 512)

print("LuckyNet feature shapes:", net_train_feats.shape, net_test_feats.shape)

# ------------------------
# 最終特徴ベクトルを結合して KNN
# PCA1 coeff (pca_dim) + AE latent (ae_latent) + LuckyNet feat (512)
# ------------------------
# NOTE: X_train_pca is (n_train, pca_dim) from before, X_test_pca likewise
final_train_feats = np.concatenate([X_train_pca, ae_train_feats, net_train_feats], axis=1)
final_test_feats  = np.concatenate([X_test_pca, ae_test_feats,  net_test_feats ], axis=1)

print("Final feature shapes:", final_train_feats.shape, final_test_feats.shape)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(final_train_feats, y_train)
pred = knn.predict(final_test_feats)
acc = accuracy_score(y_test, pred)
print(f"\nFinal Accuracy (PCA1({pca_dim}) + ResidualPCA({res_pca_dim})+AE({ae_latent}) + LuckyNet(512) + KNN): {acc:.4f}")

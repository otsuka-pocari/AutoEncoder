import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets.load_adni import load_adni2  # ★ 実データ読み込み用

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

for subject in dataset_adni:
    voxel = subject["voxel"].astype(np.float32)
    label = 0 if subject["class"] == "CN" else 1  # CN=0, AD=1
    X.append(voxel)
    y.append(label)

X = np.array(X)  # (N, 80,112,80)
y = np.array(y)
print(f"Loaded ADNI2 dataset: X={X.shape}, y={y.shape}")

# =====================================================
# 前処理：0-1 正規化（全データで）
# =====================================================
X = (X - X.min()) / (X.max() - X.min())  # 全体で正規化
N, D1, D2, D3 = X.shape
print("Data range after norm:", X.min(), X.max())

# train/test split (flatten then PCA will be fit on training only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# flatten for PCA
X_train_flat = X_train.reshape((X_train.shape[0], -1))  # (n_train, D)
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# =====================================================
# PCA（線形部分）設定
# =====================================================
pca_components = 16  # <-- 好みで調整
pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
print("Fitting PCA on flattened training data ...")
pca.fit(X_train_flat)  # fit on training only

# PCA coefficients (低次元線形表現)
train_pca_coeffs = pca.transform(X_train_flat)  # (n_train, pca_components)
test_pca_coeffs = pca.transform(X_test_flat)

# PCA 再構成（線形再構成）
train_pca_recon_flat = pca.inverse_transform(train_pca_coeffs)  # (n_train, D)
test_pca_recon_flat = pca.inverse_transform(test_pca_coeffs)

# reshape back to volume shape
train_pca_recon = train_pca_recon_flat.reshape((-1, D1, D2, D3))
test_pca_recon = test_pca_recon_flat.reshape((-1, D1, D2, D3))

# =====================================================
# 残差（original - PCA_recon）を作る
# =====================================================
train_residuals = X_train - train_pca_recon  # shape (n_train, D1, D2, D3)
test_residuals = X_test - test_pca_recon

# 残差は負になることがある（PCAは再構成誤差を持つ）
print("Residuals range:", train_residuals.min(), train_residuals.max())

# AE に入れるためにチャンネル次元を作る (N,1,D1,D2,D3)
train_residuals = np.expand_dims(train_residuals, axis=1).astype(np.float32)
test_residuals = np.expand_dims(test_residuals, axis=1).astype(np.float32)

# =====================================================
# DataLoader 作成（AE 用の残差）
# =====================================================
train_dataset_res = TensorDataset(torch.tensor(train_residuals), torch.tensor(y_train))
test_dataset_res = TensorDataset(torch.tensor(test_residuals), torch.tensor(y_test))

train_loader = DataLoader(train_dataset_res, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset_res, batch_size=4, shuffle=False)

# =====================================================
# AutoEncoder モデル定義（3D CNN） — 残差を扱うので出力活性は identity（最後の Sigmoid は外す）
# =====================================================
class ResidualAutoEncoder3D(nn.Module):
    def __init__(self, latent_dim=16):
        super(ResidualAutoEncoder3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),  # (80,112,80)→(40,56,40)
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), # →(20,28,20)
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), # →(10,14,10)
            nn.ReLU()
        )
        # flatten dim
        self.flatten_dim = 64 * 10 * 14 * 10
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1), # →(20,28,20)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1), # →(40,56,40)
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1)   # →(80,112,80)
            # NOTE: no Sigmoid here — residuals can be negative
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        z = self.fc_mu(x)
        x = self.fc_decode(z)
        x = x.view(batch_size, 64, 10, 14, 10)
        x = self.decoder(x)  # (batch,1,D1,D2,D3)
        return x, z

# =====================================================
# 学習設定
# =====================================================
residual_latent = 16  # <-- AE の潜在次元（PCA と合わせても良い）
ae = ResidualAutoEncoder3D(latent_dim=residual_latent).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=1e-4)

# =====================================================
# AE 学習ループ（残差をターゲットに学習）
# =====================================================
num_epochs = 30
save_path = "best_residual_ae.pth"
best_loss = float("inf")

print("Training Residual AE ...")
for epoch in range(num_epochs):
    ae.train()
    total_loss = 0.0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)  # 残差が入っている
        x_recon, _ = ae(x_batch)
        loss = criterion(x_recon, x_batch)  # ターゲットは残差
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Residual MSE: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(ae.state_dict(), save_path)
        print("  ✅ Residual AE saved (best so far)")

print("Residual AE training finished!")

# =====================================================
# 推論 — PCA再構成 + AE残差再構成 を足して最終再構成を得る
# =====================================================
ae.load_state_dict(torch.load(save_path))
ae.eval()

def ae_reconstruct_all(loader):
    recon_list = []
    latent_list = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            x_recon, z = ae(x_batch)
            recon_list.append(x_recon.cpu().numpy())
            latent_list.append(z.cpu().numpy())
    return np.concatenate(recon_list), np.concatenate(latent_list)

# AEで残差再構成（train/test）
train_res_recon, train_res_latent = ae_reconstruct_all(train_loader)  # shapes (n_train,1,D1,D2,D3), (n_train, latent)
test_res_recon, test_res_latent = ae_reconstruct_all(test_loader)

# reshape PCA recon and AE recon to same layout and sum
train_pca_recon_ch = np.expand_dims(train_pca_recon, axis=1)  # (n_train,1,D1,D2,D3)
test_pca_recon_ch = np.expand_dims(test_pca_recon, axis=1)

train_final_recon = train_pca_recon_ch + train_res_recon  # (n_train,1,D1,D2,D3)
test_final_recon = test_pca_recon_ch + test_res_recon

# If you want final recon in same scale as original X (0-1), verify ranges:
print("Final recon ranges (train/test):", train_final_recon.min(), train_final_recon.max(),
      test_final_recon.min(), test_final_recon.max())

# =====================================================
# 特徴抽出（KNN用）:
# PCA係数とAE潜在変数を結合して使う
# =====================================================
# train: train_pca_coeffs (n_train, pca_components), train_res_latent (n_train, residual_latent)
train_feats = np.concatenate([train_pca_coeffs, train_res_latent], axis=1)
test_feats = np.concatenate([test_pca_coeffs, test_res_latent], axis=1)
print("Feature shapes for KNN:", train_feats.shape, test_feats.shape)

# =====================================================
# KNN分類
# =====================================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_feats, y_train)
pred = knn.predict(test_feats)
acc = accuracy_score(y_test, pred)
print(f"\nKNN accuracy using [PCA coeffs + AE latent]: {acc:.4f}")

# =====================================================
# 結果出力
# =====================================================
# 例: 保存
np.savez("pca_and_residual_ae_outputs.npz",
         pca_components=pca_components,
         residual_latent=residual_latent,
         pca_mean=pca.mean_,
         pca_components_matrix=pca.components_,
         train_feats=train_feats,
         test_feats=test_feats,
         y_train=y_train,
         y_test=y_test)

print("All done. PCA + Residual AE pipeline finished.")

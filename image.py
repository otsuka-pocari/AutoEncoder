############################################################
# latent_dim を複数試して結果を表で比較
# （再構成誤差なし：latent feature のみ）
# + confusion matrix 保存
# + AD / CN 画像を表示して保存（ADNI実データ）
############################################################

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tqdm import tqdm

# -------------------------
# GPU設定
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# データ読み込み（ADNI）
# -------------------------
from datasets.load_adni import load_adni2
adni = load_adni2()

label_map = {"CN": 0, "AD": 1}
adni_filtered = [item for item in adni if item["class"] in label_map]

X_list = [item["voxel"] for item in adni_filtered]
y_list = [label_map[item["class"]] for item in adni_filtered]

X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list, dtype=int)

# チャンネル次元追加
if X_all.ndim == 4:
    X_all = X_all[:, None, ...]

# min-max normalization（per sample）
for i in range(len(X_all)):
    xmin, xmax = X_all[i].min(), X_all[i].max()
    if xmax > xmin:
        X_all[i] = (X_all[i] - xmin) / (xmax - xmin)
    else:
        X_all[i] = X_all[i] - xmin

print("X_all:", X_all.shape, "y_all:", y_all.shape)

# =========================================================
# AD / CN 画像を表示して保存（★追加部分）
# =========================================================
def save_ad_cn_images(X, y, save_path, slice_axis=2):
    """
    ADNI voxel データから AD / CN の代表スライスを保存
    X: (N, 1, D, H, W)
    y: (N,)  0=CN, 1=AD
    """

    cn_idx = np.where(y == 0)[0][0]
    ad_idx = np.where(y == 1)[0][0]

    cn_img = X[cn_idx, 0]
    ad_img = X[ad_idx, 0]

    cn_slice = cn_img.take(cn_img.shape[slice_axis] // 2, axis=slice_axis)
    ad_slice = ad_img.take(ad_img.shape[slice_axis] // 2, axis=slice_axis)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    axes[0].imshow(cn_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("CN", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(ad_slice.T, cmap="gray", origin="lower")
    axes[1].set_title("AD", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# 保存実行
os.makedirs("examples", exist_ok=True)
save_ad_cn_images(
    X_all,
    y_all,
    save_path="examples/ADNI_AD_CN_example.png",
    slice_axis=2
)
print("AD / CN example image saved.")

# -------------------------
# train / test split
# -------------------------
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.2,
    stratify=y_all,
    random_state=RANDOM_SEED
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

# -------------------------
# 3D Convolutional AutoEncoder
# -------------------------
class Conv3dAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 2, 1),
            nn.BatchNorm3d(16), nn.ReLU(True),
            nn.Conv3d(16, 32, 3, 2, 1),
            nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.BatchNorm3d(64), nn.ReLU(True),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.BatchNorm3d(128), nn.ReLU(True),
        )

        self._latent_dim = latent_dim
        self.fc_enc = None
        self.fc_dec = None

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm3d(64), nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm3d(32), nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, 2, 1, output_padding=1),
            nn.BatchNorm3d(16), nn.ReLU(True),
            nn.ConvTranspose3d(16, in_channels, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def _init_fc(self, x_sample):
        with torch.no_grad():
            x = torch.from_numpy(x_sample[None]).to(next(self.parameters()).device)
            h = self.enc(x)
            self._enc_shape = h.shape[1:]
            flat_dim = int(np.prod(self._enc_shape))
            self.fc_enc = nn.Linear(flat_dim, self._latent_dim).to(x.device)
            self.fc_dec = nn.Linear(self._latent_dim, flat_dim).to(x.device)
            self.add_module("fc_enc", self.fc_enc)
            self.add_module("fc_dec", self.fc_dec)

    def forward(self, x):
        if self.fc_enc is None:
            self._init_fc(x[0].cpu().numpy())
        h = self.enc(x).view(x.size(0), -1)
        z = self.fc_enc(h)
        h = self.fc_dec(z).view(z.size(0), *self._enc_shape)
        x_rec = self.dec_conv(h)
        return x_rec, z

# -------------------------
# Confusion matrix 保存関数
# -------------------------
def save_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(4, 4), dpi=300)
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                fontsize=12,
                path_effects=[
                    pe.Stroke(linewidth=3, foreground="white"),
                    pe.Normal()
                ]
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------
# 実験関数
# -------------------------
def run_experiment(latent_dim, epochs=10):
    model = Conv3dAutoEncoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=8)

    for _ in range(epochs):
        model.train()
        for xb, _ in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_rec, _ = model(xb)
            loss = criterion(x_rec, xb)
            loss.backward()
            optimizer.step()

    def extract(loader):
        emb, lbl = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                _, z = model(xb)
                emb.append(z.cpu().numpy())
                lbl.append(yb.numpy())
        return np.concatenate(emb), np.concatenate(lbl)

    Xtr, ytr = extract(train_loader)
    Xte, yte = extract(test_loader)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)
    pred = knn.predict(Xte)

    cm = confusion_matrix(yte, pred, labels=[0, 1])
    os.makedirs("confusion_matrices", exist_ok=True)
    save_confusion_matrix(
        cm,
        classes=["CN", "AD"],
        save_path=f"confusion_matrices/cm_latent{latent_dim}.png"
    )

    print(f"latent_dim={latent_dim}, ACC={accuracy_score(yte, pred):.4f}")

# -------------------------
# latent_dim 比較
# -------------------------
for ld in [16, 32, 64, 128, 256]:
    run_experiment(ld)

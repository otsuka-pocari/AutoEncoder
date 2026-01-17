############################################################
# latent_dim を複数試して結果を表で比較
# （再構成誤差なし：latent feature のみ）
# + confusion matrix を保存（白縁付き数字）
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
# データ読み込み
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

# min-max normalization (per sample)
for i in range(len(X_all)):
    xmin, xmax = X_all[i].min(), X_all[i].max()
    if xmax > xmin:
        X_all[i] = (X_all[i] - xmin) / (xmax - xmin)
    else:
        X_all[i] = X_all[i] - xmin

print("X_all:", X_all.shape, "y_all:", y_all.shape)

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

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_enc(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), *self._enc_shape)
        return self.dec_conv(h)

    def forward(self, x):
        if self.fc_enc is None:
            self._init_fc(x[0].cpu().numpy())
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

# -------------------------
# Confusion matrix 保存関数（白縁付き数字）
# -------------------------
def save_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(4, 4), dpi=300)
    plt.imshow(cm, cmap="Blues")

    #plt.title(title, fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
                path_effects=[
                    pe.Stroke(linewidth=3, foreground="white"),
                    pe.Normal()
                ]
            )

    plt.ylabel("True label", fontsize=11)
    plt.xlabel("Predicted label", fontsize=11)

    plt.grid(False)
    plt.gca().set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    plt.gca().grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    plt.gca().tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# -------------------------
# 実験関数
# -------------------------
def run_experiment(latent_dim, epochs=10, save_dir="confusion_matrices"):
    print(f"\n===== LATENT_DIM = {latent_dim} =====")
    os.makedirs(save_dir, exist_ok=True)

    model = Conv3dAutoEncoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)

    # ---- CAE training ----
    for _ in range(epochs):
        model.train()
        for xb, _ in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_rec, _ = model(xb)
            loss = criterion(x_rec, xb)
            loss.backward()
            optimizer.step()

    # ---- embedding extraction ----
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

    # ---- standardization ----
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # ---- kNN ----
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(Xtr, ytr)
    pred = knn.predict(Xte)

    # ---- metrics ----
    acc_all = accuracy_score(yte, pred)
    f1_macro = f1_score(yte, pred, average="macro")
    f1_each = f1_score(yte, pred, average=None, labels=[0, 1])

    acc_each = []
    for c in [0, 1]:
        idx = (yte == c)
        acc_each.append((pred[idx] == yte[idx]).mean())

    # ---- confusion matrix ----
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    save_path = os.path.join(
        save_dir, f"confusion_matrix_simple_latent{latent_dim}.png"
    )

    save_confusion_matrix(
        cm,
        classes=["CN", "AD"],
        title=f"Confusion Matrix (latent_dim={latent_dim})",
        save_path=save_path
    )

    print(f"Confusion matrix saved: {save_path}")
    print(
        f"ACC(all)={acc_all:.4f} | Macro-F1={f1_macro:.4f} | "
        f"CN: Acc={acc_each[0]:.4f}, F1={f1_each[0]:.4f} | "
        f"AD: Acc={acc_each[1]:.4f}, F1={f1_each[1]:.4f}"
    )

    return acc_all, f1_macro, acc_each[0], f1_each[0], acc_each[1], f1_each[1]

# -------------------------
# latent_dim 比較
# -------------------------
latent_list = [16, 32, 64, 128, 256]
results = []

for ld in latent_list:
    results.append([ld] + list(run_experiment(ld, epochs=10)))

# -------------------------
# 結果表示
# -------------------------
print("\n====== Summary ======")
print("LATENT | Acc(all) | F1(macro) | Acc(CN) | F1(CN) | Acc(AD) | F1(AD)")
for r in results:
    print(
        f"{r[0]:6d} | {r[1]:.4f} | {r[2]:.4f} | "
        f"{r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f}"
    )

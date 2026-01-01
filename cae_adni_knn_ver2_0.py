############################################################
# latent_dim を複数試して結果を表で比較できるコード
############################################################

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# -------------------------
# GPU設定
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# データ読み込み（リスト形式）
# -------------------------
from datasets.load_adni import load_adni2
adni = load_adni2()

label_map = {"CN":0, "AD":1}
adni_filtered = [item for item in adni if item["class"] in label_map]

X_list = [item["voxel"] for item in adni_filtered]
y_list = [label_map[item["class"]] for item in adni_filtered]

X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list, dtype=int)

# チャンネル次元追加
if X_all.ndim == 4:
    X_all = X_all[:, None, ...]

# min-max normalization per sample
for i in range(len(X_all)):
    xmin, xmax = X_all[i].min(), X_all[i].max()
    if xmax > xmin:
        X_all[i] = (X_all[i] - xmin) / (xmax - xmin)
    else:
        X_all[i] = X_all[i] - xmin

print("X_all:", X_all.shape, "y_all:", y_all.shape)

# -------------------------
# train/test split
# -------------------------
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_SEED
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

# -------------------------
# 3D CAE Model（元コードと同じ）
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
        self._flattened_size = None
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
            self._flattened_size = int(np.prod(h.shape[1:]))
            self.fc_enc = nn.Linear(self._flattened_size, self._latent_dim).to(x.device)
            self.fc_dec = nn.Linear(self._latent_dim, self._flattened_size).to(x.device)
            self.add_module("fc_enc", self.fc_enc)
            self.add_module("fc_dec", self.fc_dec)

    def encode(self, x):
        h = self.enc(x)
        b = h.shape[0]
        h = h.view(b, -1)
        z = self.fc_enc(h)
        return z

    def decode(self, z):
        b = z.shape[0]
        h_flat = self.fc_dec(z)
        h = h_flat.view(b, 128, *self._enc_dim)
        return self.dec_conv(h)

    @property
    def _enc_dim(self):
        dev = next(self.parameters()).device
        dummy = torch.zeros((1, 1, X_all.shape[2], X_all.shape[3], X_all.shape[4]), device=dev)
        with torch.no_grad():
            out = self.enc(dummy)
        return out.shape[2], out.shape[3], out.shape[4]

    def forward(self, x):
        if self.fc_enc is None:
            self._init_fc(x[0].cpu().numpy())
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

# -------------------------
# 学習 + 埋め込み抽出 + KNN をまとめた関数
# -------------------------
def run_experiment(latent_dim, epochs=60):
    print(f"\n===== LATENT_DIM = {latent_dim} =====")

    model = Conv3dAutoEncoder(in_channels=1, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)

    # ---- train CAE ----
    for epoch in range(epochs):
        model.train()
        for xb, _ in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_rec, _ = model(xb)
            loss = criterion(x_rec, xb)
            loss.backward()
            optimizer.step()

    # ---- extract embeddings ----
    def extract(loader):
        emb, re, lbl = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                x_rec, z = model(xb)
                err = ((x_rec - xb) ** 2).reshape(xb.shape[0], -1).mean(dim=1)
                emb.append(z.cpu().numpy())
                re.append(err.cpu().numpy()[:, None])
                lbl.append(yb.numpy())
        return (
            np.concatenate(emb),
            np.concatenate(re),
            np.concatenate(lbl)
        )

    tr_emb, tr_re, tr_lbl = extract(train_loader)
    te_emb, te_re, te_lbl = extract(test_loader)

    Xtr = np.concatenate([tr_emb, tr_re], axis=1)
    Xte = np.concatenate([te_emb, te_re], axis=1)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # ---- KNN ----
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(Xtr, tr_lbl)
    pred = knn.predict(Xte)

    acc = accuracy_score(te_lbl, pred)
    f1m = f1_score(te_lbl, pred, average="macro")
    print(f"ACC={acc:.4f}  F1={f1m:.4f}")

    return acc, f1m

# -------------------------
# 実験したい latent_dim をリストで指定
# -------------------------
latent_list = [16, 32, 64, 128, 256]

results = []

for ld in latent_list:
    acc, f1m = run_experiment(ld, epochs=10)  # epochs は短縮推奨
    results.append([ld, acc, f1m])

# -------------------------
# 結果を表で表示
# -------------------------
print("\n====== Summary ======")
print("LATENT_DIM | Accuracy | Macro-F1")
for ld, acc, f1m in results:
    print(f"{ld:10d} | {acc:.4f}   | {f1m:.4f}")

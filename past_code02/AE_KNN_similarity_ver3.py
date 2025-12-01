# cae_adni_knn_strlabel.py
# ADNI リスト形式対応・文字列ラベル(CN/AD) → 数値ラベル変換・再構成誤差込み KNN 分類

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

# -------------------------
# GPU設定
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# ハイパーパラメータ
# -------------------------
BATCH_SIZE = 8
EPOCHS = 60
LATENT_DIM = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42
K_NEIGHBORS = 5
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------
# 1) データ読み込み（リスト形式対応）
# -------------------------
from datasets.load_adni import load_adni2
adni = load_adni2()  # リスト形式

# --- キー確認 ---
print("First element keys:", adni[0].keys())

# --- 文字列ラベル対応（CN/AD → 0/1） ---
label_map = {"CN":0, "AD":1}  # CN=健康, AD=アルツハイマー
# CN/ADのみ抽出
adni_filtered = [item for item in adni if item["class"] in label_map]

X_list = [item["voxel"] for item in adni_filtered]
y_list = [label_map[item["class"]] for item in adni_filtered]

X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list, dtype=int)

# チャンネル次元追加
if X_all.ndim == 4:  # (N, D, H, W)
    X_all = X_all[:, None, ...]  # (N, C=1, D, H, W)

# min-max normalization per sample
for i in range(len(X_all)):
    vmin, vmax = X_all[i].min(), X_all[i].max()
    if vmax > vmin:
        X_all[i] = (X_all[i] - vmin) / (vmax - vmin)
    else:
        X_all[i] = X_all[i] - vmin

print("X_all shape:", X_all.shape, "y_all shape:", y_all.shape)

# -------------------------
# 2) train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_SEED
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# 3) 3D CAE モデル
# -------------------------
class Conv3dAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(True)
        )
        self._flattened_size = None
        self._latent_dim = latent_dim
        self.fc_enc = None
        self.fc_dec = None

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64), nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32), nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16), nn.ReLU(True),
            nn.ConvTranspose3d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
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
        h_flat = h.view(b, -1)
        z = self.fc_enc(h_flat)
        return z

    def decode(self, z):
        b = z.shape[0]
        h_flat = self.fc_dec(z)
        h = h_flat.view(b, 128, *self._get_enc_dim())
        x_rec = self.dec_conv(h)
        return x_rec

    def forward(self, x):
        if self.fc_enc is None:
            self._init_fc(x[0].cpu().numpy())
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def _get_enc_dim(self):
        dev = next(self.parameters()).device
        dummy = torch.zeros((1, 1, X_all.shape[2], X_all.shape[3], X_all.shape[4]), device=dev)
        with torch.no_grad():
            out = self.enc(dummy)
        return out.shape[2], out.shape[3], out.shape[4]

# -------------------------
# 4) モデル訓練
# -------------------------
model = Conv3dAutoEncoder(in_channels=1, latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()

best_loss = 1e9
patience, patience_cnt = 10, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for xb, _ in pbar:
        xb = xb.to(device)
        optimizer.zero_grad()
        x_rec, _ = model(xb)
        loss = criterion(x_rec, xb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        pbar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(train_loader.dataset)

    # validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            x_rec, _ = model(xb)
            val_loss += criterion(x_rec, xb).item() * xb.size(0)
    val_loss /= len(test_loader.dataset)
    print(f"Epoch {epoch}: TrainLoss={epoch_loss:.6f} ValLoss={val_loss:.6f}")

    if val_loss < best_loss - 1e-6:
        best_loss = val_loss
        patience_cnt = 0
        torch.save(model.state_dict(), "best_cae.pth")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(torch.load("best_cae.pth", map_location=device))
model.eval()
print("Training finished. Best val loss:", best_loss)

# -------------------------
# 5) 埋め込みと再構成誤差抽出
# -------------------------
def extract_embeddings_and_recon(loader):
    embeddings, rec_errors, labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            x_rec, z = model(xb)
            embeddings.append(z.cpu().numpy())
            mse = ((x_rec - xb) ** 2).reshape(xb.shape[0], -1).mean(dim=1).cpu().numpy()
            rec_errors.append(mse[:, None])
            labels.append(yb.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    rec_errors = np.concatenate(rec_errors, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, rec_errors, labels

train_emb, train_re, train_lbl = extract_embeddings_and_recon(train_loader)
test_emb, test_re, test_lbl = extract_embeddings_and_recon(test_loader)

# -------------------------
# 6) 特徴量結合 & 標準化
# -------------------------
X_train_feat = np.concatenate([train_emb, train_re], axis=1)
X_test_feat  = np.concatenate([test_emb, test_re], axis=1)

scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_test_feat  = scaler.transform(X_test_feat)

# -------------------------
# 7) K-NN 分類
# -------------------------
knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, n_jobs=-1)
knn.fit(X_train_feat, train_lbl)
pred = knn.predict(X_test_feat)

acc = accuracy_score(test_lbl, pred)
f1_macro = f1_score(test_lbl, pred, average="macro")
report = classification_report(test_lbl, pred)

print("K-NN Results:")
print("Accuracy:", acc)
print("Macro F1:", f1_macro)
print("Classification report:\n", report)

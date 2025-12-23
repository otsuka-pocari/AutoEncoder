############################################################
# 3D-CAE + Metric Learning (Euclidean / Mahalanobis / LMNN / NCA)
# AD/CN CBIR performance comparison (FULL & FIXED VERSION)
############################################################

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import LedoitWolf

from metric_learn import LMNN, NCA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
adni = [x for x in adni if x["class"] in label_map]

X = np.array([x["voxel"] for x in adni], dtype=np.float32)
y = np.array([label_map[x["class"]] for x in adni], dtype=int)

# channel dim
if X.ndim == 4:
    X = X[:, None, ...]

# per-sample min-max normalization
for i in range(len(X)):
    xmin, xmax = X[i].min(), X[i].max()
    X[i] = (X[i] - xmin) / (xmax - xmin + 1e-8)

print("X:", X.shape, "y:", y.shape)

# -------------------------
# train / test split
# -------------------------
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
test_ds  = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

# -------------------------
# 3D Convolutional Autoencoder
# -------------------------
class Conv3dAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, 3, 2, 1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, 3, 2, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 3, 2, 1), nn.BatchNorm3d(128), nn.ReLU(),
        )

        self.latent_dim = latent_dim
        self.fc_enc = None
        self.fc_dec = None

        self.dec = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 3, 2, 1, output_padding=1),
            nn.BatchNorm3d(16), nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def _init_fc(self, x):
        with torch.no_grad():
            h = self.enc(x)
            self.shape = h.shape[1:]
            flat = int(np.prod(self.shape))
            self.fc_enc = nn.Linear(flat, self.latent_dim).to(x.device)
            self.fc_dec = nn.Linear(self.latent_dim, flat).to(x.device)

    def forward(self, x):
        if self.fc_enc is None:
            self._init_fc(x)
        h = self.enc(x).view(x.size(0), -1)
        z = self.fc_enc(h)
        h = self.fc_dec(z).view(x.size(0), *self.shape)
        xrec = self.dec(h)
        return xrec, z

# -------------------------
# Metric Learning + kNN
# -------------------------
def knn_with_metric(Xtr, Xte, ytr, yte, method):

    if method == "euclidean":
        knn = KNeighborsClassifier(n_neighbors=5)

    elif method == "mahalanobis":
        cov = LedoitWolf().fit(Xtr).covariance_
        VI = np.linalg.pinv(cov)   # ★ 修正点：inverse covariance
        knn = KNeighborsClassifier(
            n_neighbors=5,
            metric="mahalanobis",
            metric_params={"VI": VI}
        )

    elif method == "lmnn":
        lmnn = LMNN(k=5, max_iter=200)
        lmnn.fit(Xtr, ytr)
        Xtr = lmnn.transform(Xtr)
        Xte = lmnn.transform(Xte)
        knn = KNeighborsClassifier(n_neighbors=5)

    elif method == "nca":
        nca = NCA(max_iter=200, random_state=42)
        nca.fit(Xtr, ytr)
        Xtr = nca.transform(Xtr)
        Xte = nca.transform(Xte)
        knn = KNeighborsClassifier(n_neighbors=5)

    else:
        raise ValueError("Unknown metric method")

    knn.fit(Xtr, ytr)
    pred = knn.predict(Xte)

    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    return acc, f1m

# -------------------------
# 実験
# -------------------------
def run_experiment(latent_dim, epochs=10):
    print(f"\n===== latent_dim = {latent_dim} =====")

    model = Conv3dAutoEncoder(latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    # ---- CAE training ----
    for _ in range(epochs):
        model.train()
        for xb, _ in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            xrec, _ = model(xb)
            loss = loss_fn(xrec, xb)
            loss.backward()
            opt.step()

    # ---- embedding extraction ----
    def extract(ds):
        loader = DataLoader(ds, batch_size=8)
        emb, err, lbl = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                xrec, z = model(xb)
                e = ((xrec - xb) ** 2).view(xb.size(0), -1).mean(1)
                emb.append(z.cpu().numpy())
                err.append(e.cpu().numpy()[:, None])
                lbl.append(yb.numpy())
        return np.vstack(emb), np.vstack(err), np.hstack(lbl)

    Ztr, Etr, ytr = extract(train_ds)
    Zte, Ete, yte = extract(test_ds)

    Xtr = np.hstack([Ztr, Etr])
    Xte = np.hstack([Zte, Ete])

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    for m in ["euclidean", "mahalanobis", "lmnn", "nca"]:
        acc, f1m = knn_with_metric(Xtr, Xte, ytr, yte, m)
        print(f"{m:12s} | ACC={acc:.4f}  F1={f1m:.4f}")

# -------------------------
# 実行
# -------------------------
for ld in [16, 32, 64, 128, 256]:
    run_experiment(ld)

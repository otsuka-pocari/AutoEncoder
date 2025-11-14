import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets.load_adni import load_adni2
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

for subject in dataset_adni:
    voxel = subject["voxel"]
    label = 0 if subject["class"] == "CN" else 1
    X.append(voxel)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Loaded ADNI2 dataset: X={X.shape}, y={y.shape}")

# =====================================================
# 前処理
# =====================================================
X = (X - X.min()) / (X.max() - X.min())  # 正規化
X = np.expand_dims(X, axis=1)  # (N,1,80,112,80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# =====================================================
# LuckyNet (3D CNN)
# =====================================================
class LuckyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)

        self.conv1 = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, padding=1)

        # ADNI2 shape: 80 × 112 × 80
        # After pool → (40,56,40)
        # After pool → (20,28,20)
        # After pool → (10,14,10)
        self.flatten_dim = 32 * 10 * 14 * 10  # 44800

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.25)

        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(32)
        self.batchnorm1d = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(x)   # (1,80,112,80) → (1,40,56,40)

        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = F.relu(self.batchnorm3d2(self.conv2(x)))

        x = self.pool(x)   # (3,40,56,40) → (3,20,28,20)
        x = F.relu(self.batchnorm3d3(self.conv3(x)))
        x = F.relu(self.batchnorm3d4(self.conv4(x)))

        x = self.pool(x)   # (32,20,28,20) → (32,10,14,10)

        x = x.view(-1, self.flatten_dim)

        x = self.dropout(x)
        feat = F.relu(self.batchnorm1d(self.fc1(x)))  # ★特徴ベクトル（512次元）

        x = self.dropout(feat)
        out = self.fc2(x)  # 2クラス分類（使わない）

        return out, feat  # ★KNN 用に feat を返す

# =====================================================
# LuckyNet 学習
# =====================================================
model = LuckyNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20
save_path = "luckynet_best.pth"
best_loss = float("inf")

print("\nTraining LuckyNet...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred, _ = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print("  ✅ Model saved (best so far)")

print("LuckyNet training finished!\n")

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
            _, z = model(x_batch)   # ★ 512 次元特徴
            feats.append(z.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.concatenate(feats), np.concatenate(labels)

train_feats, train_labels = extract_features(train_loader)
test_feats, test_labels = extract_features(test_loader)

print("Feature extraction done:", train_feats.shape, test_feats.shape)

# =====================================================
# KNN分類
# =====================================================
print("\nTraining KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_feats, train_labels)

pred = knn.predict(test_feats)
acc = accuracy_score(test_labels, pred)

print(f"\n🔥 KNN accuracy: {acc:.4f}\n")
print("All done.")

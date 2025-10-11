import glob
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# ===================================
# Import your local modules
# ===================================
from datasets.load_adni import load_adni2
from utils.data_class import BrainDataset
from models.models import DeepCNNAutoEncoder3D

# ===================================
# Class mapping and random seed
# ===================================
CLASS_MAP = {"CN": 0, "AD": 1}
SEED_VALUE = 0

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(SEED_VALUE)

# ===================================
# Load dataset
# ===================================
dataset = load_adni2(classes=["CN", "AD"], size="half", unique=True, mni=False, strength=["3.0"])
print("Dataset size:", len(dataset))

# ===================================
# Extract voxel and label
# ===================================
pids = []
voxels = np.zeros((len(dataset), 80, 112, 80))
labels = np.zeros(len(dataset))
for i in tqdm(range(len(dataset))):
    pids.append(dataset[i]["pid"])
    voxels[i] = dataset[i]["voxel"]
    labels[i] = CLASS_MAP[dataset[i]["class"]]
pids = np.array(pids)

# ===================================
# Split train / val / test = 8:1:1
# ===================================
outer_split = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED_VALUE)
train_val_idx, test_idx = list(outer_split.split(voxels, labels, groups=pids))[0]

train_val_voxels = voxels[train_val_idx]
train_val_labels = labels[train_val_idx]
train_val_pids = pids[train_val_idx]

test_voxels = voxels[test_idx]
test_labels = labels[test_idx]

inner_split = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED_VALUE)
train_idx, val_idx = list(inner_split.split(train_val_voxels, train_val_labels, groups=train_val_pids))[0]

train_voxels = train_val_voxels[train_idx]
val_voxels = train_val_voxels[val_idx]
train_labels = train_val_labels[train_idx]
val_labels = train_val_labels[val_idx]

print(f"Train size: {len(train_voxels)}")
print(f"Val size: {len(val_voxels)}")
print(f"Test size: {len(test_voxels)}")

# ===================================
# Dataset & DataLoader
# ===================================
train_set = BrainDataset(train_voxels, train_labels)
val_set = BrainDataset(val_voxels, val_labels)
test_set = BrainDataset(test_voxels, test_labels)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# ===================================
# Training setup
# ===================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 64  # 固定して良いが、後で変えてもOK
model = DeepCNNAutoEncoder3D(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===================================
# Train with validation monitoring
# ===================================
best_val_loss = float("inf")
output_dir = "reconstructed_image_DeepCNN_best"
os.makedirs(output_dir, exist_ok=True)

for epoch in range(1, 101):
    model.train()
    total_train_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    count = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            total_val_loss += loss.item() * x.size(0)
            count += x.size(0)
    avg_val_loss = total_val_loss / count

    print(f"Epoch [{epoch}/100] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print(f"✅ Best model saved at epoch {epoch} (Val Loss: {best_val_loss:.6f})")

# ===================================
# Test evaluation (after training)
# ===================================
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
model.eval()

total_test_loss = 0
count = 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        x_hat = model(x)
        loss = criterion(x_hat, x)
        total_test_loss += loss.item() * x.size(0)
        count += x.size(0)
avg_test_loss = total_test_loss / count

print(f"\n🎯 Final Test Reconstruction Error (MSE): {avg_test_loss:.6f}")

# ===================================
# Save reconstruction example
# ===================================
test_imgs = next(iter(test_loader))[0][:8].to(device)
with torch.no_grad():
    reconstructed = model(test_imgs)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    mid_slice = test_imgs[i].shape[2] // 2
    axes[0, i].imshow(test_imgs[i, 0, :, mid_slice, :].cpu().numpy(), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Original", fontsize=8)
    axes[1, i].imshow(reconstructed[i, 0, :, mid_slice, :].cpu().numpy(), cmap="gray")
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstructed", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "reconstructed_best_model.png"))
plt.close()

print("\n✅ Done: Best model and reconstruction image saved.")

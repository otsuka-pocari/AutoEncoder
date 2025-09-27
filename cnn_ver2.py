import glob
import os.path as osp
import pickle
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt

import datasets.dataset as dataset
from datasets.load_adni import load_adni2
from models.models import ShallowCNNAutoEncoder3D, CNNAutoEncoder3D, DeepCNNAutoEncoder3D
from utils.data_class import BrainDataset
import torchio as tio
from tqdm import tqdm
import os
from sklearn.model_selection import StratifiedGroupKFold

# class mapping
CLASS_MAP = {
    "CN": 0,
    "AD": 1,
}

# reproducibility
SEED_VALUE = 0

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

fix_seed(SEED_VALUE)

# load dataset
dataset = load_adni2(classes=["CN", "AD"], size="half", unique=True, mni=False, strength=["3.0"])
print("Dataset size:", len(dataset))

# extract voxel and label
pids = []
voxels = np.zeros((len(dataset), 80, 112, 80))
labels = np.zeros(len(dataset))
for i in tqdm(range(len(dataset))):
    pids.append(dataset[i]["pid"])
    voxels[i] = dataset[i]["voxel"]
    labels[i] = CLASS_MAP[dataset[i]["class"]]
pids = np.array(pids)

# ===================================
# Split train/val/test (8:1:1, no patient overlap)
# ===================================
gss_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED_VALUE)

# outer split: train_val(90%) vs test(10%)
train_val_idx, test_idx = list(gss_outer.split(voxels, labels, groups=pids))[0]

train_val_voxels = voxels[train_val_idx]
test_voxels = voxels[test_idx]
train_val_labels = labels[train_val_idx]
test_labels = labels[test_idx]
train_val_pids = pids[train_val_idx]
test_pids = pids[test_idx]

# inner split: train(80%) vs val(10%)
gss_inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED_VALUE)
train_idx, val_idx = list(gss_inner.split(train_val_voxels, train_val_labels, groups=train_val_pids))[0]

train_voxels = train_val_voxels[train_idx]
val_voxels = train_val_voxels[val_idx]
train_labels = train_val_labels[train_idx]
val_labels = train_val_labels[val_idx]

print("Train size =", len(train_voxels))
print("Validation size =", len(val_voxels))
print("Test size =", len(test_voxels))

# dataset
train_set = BrainDataset(train_voxels, train_labels)
val_set = BrainDataset(val_voxels, val_labels)
test_set = BrainDataset(test_voxels, test_labels)

# dataloader
train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False)

# output dir
os.makedirs('reconstructed_image_ver2_DeepCNN_100epoch', exist_ok=True)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# latent dimensions to test
latent_dims = [2, 8, 16, 32, 64, 128, 256]
reconstruction_errors_test = []

# ===================================
# Training loop & evaluation
# ===================================
for latent_dim in latent_dims:
    print(f"\n=== Latent Dimension: {latent_dim} ===")

    model = DeepCNNAutoEncoder3D(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None

    # training
    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, _ in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)

        # validation6s
        model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for x, _ in val_dataloader:
                x = x.to(device)
                x_hat = model(x)
                loss = criterion(x_hat, x)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
        avg_val_loss = total_loss / count

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # モデル保存（valが改善したときだけ）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    # best modelで評価
    model.load_state_dict(best_model_state)

    # test loss
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
    avg_loss_test = total_loss / count
    reconstruction_errors_test.append(avg_loss_test)
    print(f"Test Reconstruction Error (MSE): {avg_loss_test:.6f}")

    # save reconstruction images (from test set)
    test_imgs = next(iter(test_dataloader))[0][:8].to(device)
    with torch.no_grad():
        reconstructed = model(test_imgs)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        mid_slice = test_imgs[i].shape[2] // 2
        axes[0, i].imshow(test_imgs[i, 0, :, mid_slice, :].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original", fontsize=8)
        axes[1, i].imshow(reconstructed[i, 0, :, mid_slice, :].cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed", fontsize=8)
    plt.tight_layout()
    plt.savefig(f'reconstructed_image_ver2_DeepCNN_100epoch/reconstructed_grid_latent_{latent_dim}.png')
    plt.close()

# plot reconstruction error (test only)
plt.figure(figsize=(8, 5))
plt.plot(latent_dims, reconstruction_errors_test, marker='o', linestyle='-', label="Test")
plt.title("Latent Dimension vs Reconstruction Error (Test Only)")
plt.xlabel("Latent Dimension")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.savefig("reconstructed_image_ver2_DeepCNN_100epoch/reconstruction_error_plot_test_only.png")
plt.close()

print("\n✅ Done: Reconstruction images and test-only error plot saved.")

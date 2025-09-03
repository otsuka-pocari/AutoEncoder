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
from models.models import CNN, CNNAutoEncoder3D
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

# split train/val (no patient overlap)
gss = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED_VALUE)
train_idx, val_idx = list(gss.split(voxels, labels, groups=pids))[0]
train_voxels = voxels[train_idx]
val_voxels = voxels[val_idx]
train_labels = labels[train_idx]
val_labels = labels[val_idx]

# dataset
train_set = BrainDataset(train_voxels, train_labels)
val_set = BrainDataset(val_voxels, val_labels)

print("Train size =", len(train_set))
print("Validation size =", len(val_set))

# dataloader
train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=16, shuffle=False)

# output dir
os.makedirs('reconstructed_image_CNN_100epoch', exist_ok=True)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# latent dimensions to test
latent_dims = [2, 8, 16, 32, 64, 128, 256]
reconstruction_errors = []

# ===================================
# Training loop & evaluation
# ===================================
for latent_dim in latent_dims:
    print(f"\n=== Latent Dimension: {latent_dim} ===")

    model = CNNAutoEncoder3D(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # training
    model.train()
    for epoch in range(100):
        total_loss = 0
        for x, _ in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_dataloader):.4f}")

    # validation loss
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
    avg_loss = total_loss / count
    reconstruction_errors.append(avg_loss)
    print(f"Validation Reconstruction Error (MSE): {avg_loss:.6f}")

    # save reconstruction images
    test_imgs = next(iter(val_dataloader))[0][:8].to(device)
    with torch.no_grad():
        reconstructed = model(test_imgs)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        mid_slice = test_imgs[i].shape[2] // 2
        axes[0, i].imshow(test_imgs[i, 0, mid_slice, :, :].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original", fontsize=8)
        axes[1, i].imshow(reconstructed[i, 0, mid_slice, :, :].cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed", fontsize=8)
    plt.tight_layout()
    plt.savefig(f'reconstructed_image_CNN_100epoch/reconstructed_grid_latent_{latent_dim}.png')
    plt.close()

# plot reconstruction error
plt.figure(figsize=(8, 5))
plt.plot(latent_dims, reconstruction_errors, marker='o', linestyle='-')
plt.title("Latent Dimension vs Reconstruction Error (3D CNN AE)")
plt.xlabel("Latent Dimension")
plt.ylabel("Reconstruction Error (MSE)")
plt.grid(True)
plt.savefig("reconstructed_image_CNN_100epoch/reconstruction_error_plot.png")
plt.close()

print("\n✅ Done: Reconstruction images and error plot saved.")

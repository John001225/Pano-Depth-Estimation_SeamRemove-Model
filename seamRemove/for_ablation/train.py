import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import PanoDepthDataset
from model import get_model
from losses import total_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PanoDepth Seam-Aware Training with Ablation Study')
    parser.add_argument('--use_l1', action='store_true', help='Use L1 loss')
    parser.add_argument('--use_grad', action='store_true', help='Use gradient loss')
    parser.add_argument('--use_lap', action='store_true', help='Use laplacian loss')
    parser.add_argument('--exp_name', type=str, default=None, help='Custom experiment name')
    return parser.parse_args()

args = parse_args()

USE_L1 = args.use_l1
USE_GRAD = args.use_grad
USE_LAP = args.use_lap

# 自動產生實驗名稱
loss_flags = f"L1{int(USE_L1)}_Grad{int(USE_GRAD)}_Lap{int(USE_LAP)}"
exp_name = args.exp_name if args.exp_name else f"ablation_{loss_flags}"

# log 與 checkpoint 路徑
log_dir = f"runs/{exp_name}"
save_path = f"checkpoints/{exp_name}"
os.makedirs(save_path, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
batch_size = 2
epochs = 100
lr = 1e-4
image_size = (1024, 2048)
root = '../dataset'
# save_path = 'checkpoints_stitched_new'
# os.makedirs(save_path, exist_ok=True)

# writer = SummaryWriter(log_dir='runs/seam_removal_stitched_new')


# Transform
transform_rgb = T.Compose([
    T.Resize(image_size),
    T.ToTensor()
])
transform_depth = T.Compose([
    T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor()
])

# Dataset & Dataloader
train_set = PanoDepthDataset(root, split='train', transform_rgb=transform_rgb, transform_depth=transform_depth)
val_set = PanoDepthDataset(root, split='val', transform_rgb=transform_rgb, transform_depth=transform_depth)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=1)

# Model
model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_val_loss = float('inf')

# Training loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training {exp_name}", dynamic_ncols=True)

    for input_tensor, gt in train_pbar:
        input_tensor, gt = input_tensor.to(device), gt.to(device)

        pred = model(input_tensor)
        # dont use, because gt have missing area with value = 0
        pred = torch.clamp(pred, min=0.1, max=10.0)
        # gt = torch.clamp(gt, min=0.1, max=10.0)

        # loss = total_loss(pred, gt)
        loss = total_loss(pred, gt, use_l1=USE_L1, use_grad=USE_GRAD, use_lap=USE_LAP)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    tqdm.write(f"[{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}")
    writer.add_scalar('Loss/train', avg_train_loss, epoch)

    # Validation
    model.eval()
    val_loss = 0.0
    val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation {exp_name}", dynamic_ncols=True)
    with torch.no_grad():
        for input_tensor, gt in val_pbar:
            input_tensor, gt = input_tensor.to(device), gt.to(device)

            pred = model(input_tensor)
            # dont use, because gt have missing area with value = 0
            pred = torch.clamp(pred, min=0.1, max=10.0)
            # gt = torch.clamp(gt, min=0.1, max=10.0)

            # loss_val = total_loss(pred, gt).item()
            loss_val = total_loss(pred, gt, use_l1=USE_L1, use_grad=USE_GRAD, use_lap=USE_LAP).item()

            val_loss += loss_val
            val_pbar.set_postfix(batch_loss=f"{loss_val:.4f}")

    avg_val_loss = val_loss / len(val_loader)
    tqdm.write(f"         Val Loss: {avg_val_loss:.4f}")
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_file = os.path.join(save_path, f"best_model_epoch{epoch+1}_loss{avg_val_loss:.4f}_{timestamp}.pt")
        torch.save(model.state_dict(), save_file)
        tqdm.write(f"✅ Saved new best model to: {save_file}")

with open(os.path.join(save_path, 'config.txt'), 'w') as f:
    f.write(f"use_l1: {USE_L1}\n")
    f.write(f"use_grad: {USE_GRAD}\n")
    f.write(f"use_lap: {USE_LAP}\n")

writer.close()

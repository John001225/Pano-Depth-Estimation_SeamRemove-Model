import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import PanoDepthDataset
from model import get_model
from losses import total_loss
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
batch_size = 1
image_size = (1024, 2048)
root = './dataset'
type = 'stitched'
checkpoint_path = f'checkpoints_{type}/best_model_epoch41_loss0.2100_20250531-070334.pt'  # 修改為你的 checkpoint
save_dir = f'predictions_{type}_16bit'
os.makedirs(save_dir, exist_ok=True)

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
test_set = PanoDepthDataset(root, split='test', transform_rgb=transform_rgb, transform_depth=transform_depth, type=type)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Model
model = get_model().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Metrics accumulator
rmse, mae, absrel, rmse_log = 0.0, 0.0, 0.0, 0.0
delta1, delta2, delta3 = 0.0, 0.0, 0.0
eps = 1e-6
total_test_loss = 0.0

# Evaluation loop
test_pbar = tqdm(test_loader, desc=f"Evaluating {type}", dynamic_ncols=True)
with torch.no_grad():
    for idx, (input_tensor, gt) in enumerate(test_pbar):
        input_tensor, gt = input_tensor.to(device), gt.to(device)

        pred = model(input_tensor)
        pred = torch.clamp(pred, min=0.1, max=10.0)  # Clamp to valid range
        gt = gt.clamp(min=0.1, max=10.0)

        # Compute loss
        loss = total_loss(pred, gt).item()
        total_test_loss += loss

        # Save 16-bit prediction
        pred_np = pred.squeeze().cpu().numpy()  # [H, W]
        pred_16bit = (pred_np * 1000).astype(np.uint16)  # 保存為毫米 (x1000)
        filename = f"{idx+1:04d}_seamless.png"
        Image.fromarray(pred_16bit).save(os.path.join(save_dir, filename))

#         # Mask out invalid ground truth (e.g. == 0)
#         gt_np = gt.squeeze().cpu().numpy()
#         mask = gt_np > 0
#         if mask.sum() == 0:
#             continue

#         pred_masked = pred_np[mask]
#         gt_masked = gt_np[mask]

#         # Metrics
#         abs_diff = np.abs(pred_masked - gt_masked)
#         mae += abs_diff.mean()
#         rmse += np.sqrt((abs_diff ** 2).mean())

#         absrel += (abs_diff / gt_masked).mean()
#         rmse_log += np.sqrt((np.log(pred_masked + eps) - np.log(gt_masked + eps))**2).mean()

#         ratio = np.maximum(pred_masked / gt_masked, gt_masked / pred_masked)
#         delta1 += (ratio < 1.25).mean()
#         delta2 += (ratio < 1.25 ** 2).mean()
#         delta3 += (ratio < 1.25 ** 3).mean()

# # Normalize metrics
# N = len(test_loader)
# print("\n📊 Evaluation Metrics (Test Set):")
# print(f"🔹 MAE      : {mae / N:.4f}")
# print(f"🔹 RMSE     : {rmse / N:.4f}")
# print(f"🔹 AbsRel   : {absrel / N:.4f}")
# print(f"🔹 RMSE_log : {rmse_log / N:.4f}")
# print(f"🔹 δ1       : {delta1 / N:.4f}")
# print(f"🔹 δ2       : {delta2 / N:.4f}")
# print(f"🔹 δ3       : {delta3 / N:.4f}")
# print(f"🔹 Total Loss (avg): {total_test_loss / N:.4f}")

import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from dataset import PanoDepthDataset
from model import get_model
from losses import total_loss
from tqdm import tqdm
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
batch_size = 1
image_size = (1024, 2048)
root = './dataset'
type = 'stitched'
checkpoint_path = f'checkpoints_{type}_new/best_model_epoch94_loss0.2009_20250605-181129.pt'  # ← 改成你的檔案
# type2s = ['depthAnything_metric_raw_5_6_5_fold_padding_6', 'depthAnythingV2_metric_raw_5_6_5_fold_padding_6', 'leres_5_6_5_fold_padding_6', 'zoe_raw_5_6_5_fold_padding_6']
# type2s = ['depthAnythingV2_metric_raw_5_6_5_fold_padding_6', 'leres_5_6_5_fold_padding_6', 'depthAnything_metric_raw_5_6_5_fold', 'depthAnything_metric_raw_6_fold_padding', 'depthAnything_metric_raw_fold5', 'depthAnything_metric_raw_fold6']
# type2s = ["depthAnything_metric_raw_3_fold", "depthAnything_metric_raw_4_fold", "depthAnything_metric_raw_3_6_3_fold", "depthAnything_metric_raw_4_6_4_fold"
#           , "depthAnything_metric_raw_6_fold_padding_15", "depthAnything_metric_raw_6_fold_padding_30"]
# type2s = ['depthAnything_metric_raw_5_6_5_fold_padding_6', 'depthAnythingV2_metric_raw_5_6_5_fold_padding_6', 'leres_5_6_5_fold_padding_6', 'zoe_raw_5_6_5_fold_padding_6',
#           "depthAnything_metric_raw_3_fold", "depthAnything_metric_raw_4_fold", "depthAnything_metric_raw_3_6_3_fold", "depthAnything_metric_raw_4_6_4_fold", 
#           "depthAnything_metric_raw_6_fold_padding_15", "depthAnything_metric_raw_6_fold_padding_30", 'depthAnything_metric_raw_5_6_5_fold', 
#           'depthAnything_metric_raw_6_fold_padding', 'depthAnything_metric_raw_fold5', 'depthAnything_metric_raw_fold6']
# type2s = ['depthAnything_metric_raw_5_6_5_fold_padding_6']
type2s = ['depthAnything_metric_raw_5_6_5_fold', 'depthAnything_metric_raw_5_6_5_fold_padding_6', 
          'depthAnything_metric_raw_fold6', 'depthAnything_metric_raw_6_fold_padding', 'depthAnything_metric_raw_fold5']

for type2 in type2s:
    type2dir = f'output/{type2}'
    os.makedirs(type2dir, exist_ok=True)
    save_dir = f'{type2dir}/result_{type}_seamless'
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
    test_set = PanoDepthDataset(root, split='test', transform_rgb=transform_rgb, transform_depth=transform_depth, type=type, type2=type2)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model
    model = get_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Evaluate
    total_test_loss = 0.0
    test_pbar = tqdm(test_loader, desc=f"Evaluating on Test Set {type2} {type}", dynamic_ncols=True)
    start = time.time()

    with torch.no_grad():
        for idx, (input_tensor, gt) in enumerate(test_pbar):
            input_tensor, gt = input_tensor.to(device), gt.to(device)

            pred = model(input_tensor)
            pred = torch.clamp(pred, min=0.1, max=10.0)

            loss = total_loss(pred, gt).item()
            total_test_loss += loss
            test_pbar.set_postfix(batch_loss=f"{loss:.4f}")

            # -------- Save prediction as PNG --------
            pred_vis = pred.squeeze(0).squeeze(0).cpu()  # shape: [H, W]
            pred_vis = pred_vis / pred_vis.max()         # Normalize to 0~1 for saving
            pred_vis = (pred_vis * 255).byte()           # Convert to 0~255
            pred_pil = Image.fromarray(pred_vis.numpy(), mode='L')

            filename = f"{idx+1:04d}_seamless.png"
            pred_pil.save(os.path.join(save_dir, filename))

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"\n✅ Average Test Loss: {avg_test_loss:.4f}")
    end = time.time()
    txt_file = os.path.join(save_dir, f'{type2}_seamremove.txt')
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"執行時間: {end-start:.4f} 秒\n")

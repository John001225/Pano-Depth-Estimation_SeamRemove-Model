import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import cv2

class PanoDepthDataset(Dataset):
    def __init__(self, root, split='train', transform_rgb=None, transform_depth=None, max_depth_meters=10.0, type='stitched', type2=''):
        self.rgb_dir = os.path.join(root, split, 'rgb')
        if type2 =='':
            self.input_depth_dir = os.path.join(root, split, f'input_{type}')
        else:
            self.input_depth_dir = os.path.join(root, split, type2, f'input_{type}')
        self.gt_depth_dir = os.path.join(root, split, 'gt')

        self.rgb_files = sorted(os.listdir(self.rgb_dir))
        self.input_files = sorted(os.listdir(self.input_depth_dir))
        self.gt_files = sorted(os.listdir(self.gt_depth_dir))

        assert len(self.rgb_files) == len(self.input_files) == len(self.gt_files), \
            "三個資料夾中的檔案數量不同，無法對應"

        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.max_depth_meters = max_depth_meters

        self.type2 = type2

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # 路徑
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        input_depth_path = os.path.join(self.input_depth_dir, self.input_files[idx])
        gt_depth_path = os.path.join(self.gt_depth_dir, self.gt_files[idx])

        # RGB
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = self.transform_rgb(rgb)

        # 深度圖用 cv2 讀取為 16-bit
        # input_depth = cv2.imread(input_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 4000.0
        depth_raw = cv2.imread(input_depth_path, cv2.IMREAD_UNCHANGED)

        if depth_raw.dtype == np.uint8:
            input_depth = depth_raw.astype(np.float32) / 255.0 * self.max_depth_meters
            # print('8bit')
        elif depth_raw.dtype == np.uint16:
            if self.type2 == 'depthAnythingV2_metric_raw_5_6_5_fold_padding_6' or self.type2 == 'leres_5_6_5_fold_padding_6':
                input_depth = depth_raw.astype(np.float32) / 65535.0 * self.max_depth_meters
            else:
                input_depth = depth_raw.astype(np.float32) / 4000.0
            # print('16bit')
        else:
            raise ValueError(f"Unsupported depth image bit depth: {depth_raw.dtype}")

        gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 4000.0

        # Clamp 超過最大深度的部分（可選）
        input_depth[input_depth > self.max_depth_meters + 1] = self.max_depth_meters + 1
        gt_depth[gt_depth > self.max_depth_meters + 1] = self.max_depth_meters + 1

        # Resize to (H, W)
        input_depth = cv2.resize(input_depth, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)
        gt_depth = cv2.resize(gt_depth, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)

        # 轉成 Tensor
        input_depth = torch.from_numpy(input_depth).unsqueeze(0)  # [1, H, W]
        gt_depth = torch.from_numpy(gt_depth).unsqueeze(0)        # [1, H, W]

        # Concatenate
        input_tensor = torch.cat([rgb, input_depth], dim=0)       # [4, H, W]

        return input_tensor, gt_depth

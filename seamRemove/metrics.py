import os
import numpy as np
import cv2
from tqdm import tqdm

def load_depth(path):
    """讀取深度圖，並依照 bit 轉換成浮點數"""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"讀取失敗: {path}")
    if depth.dtype == np.uint8:
        depth = depth.astype(np.float64) / 255.0
    elif depth.dtype == np.uint16:
        depth = depth.astype(np.float64) / 65535.0
    else:
        depth = depth.astype(np.float64)
    return depth

def compute_metrics(gt, pred, align_way=0, cap_depth=True):
    """計算單張深度圖的 metrics"""
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]

    if align_way == 1:  
        shift = np.median(gt) - np.median(pred)
        pred = pred + shift
    elif align_way == 2:  
        scale = np.median(gt) / np.median(pred)
        pred = pred * scale

    to_Matterport = np.float64(65535.0) / 4000.0  #0~1 to meter (0~65535 divide by 4000 to get meter)
    depth_max = 10.0 / to_Matterport
    if cap_depth:
        pred = np.clip(pred, 0, depth_max)
        # gt = np.clip(gt, 0, depth_max)

    # 誤差計算（向量化）
    diff = pred - gt
    abs_rel = np.mean(np.abs(diff) / gt)
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    rmselog = np.sqrt(np.mean((np.log10(gt) - np.log10(pred)) ** 2))

    # δ 指標
    ratio = np.maximum(gt / pred, pred / gt)
    d1 = np.mean(ratio < 1.25)
    d2 = np.mean(ratio < 1.25 ** 2)
    d3 = np.mean(ratio < 1.25 ** 3)

    return rmse, mae, abs_rel, rmselog, d1, d2, d3

def evaluate_folder(gt_dir, pred_dir, output_txt, align_way=0, cap_depth=True):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))])

    if len(gt_files) != len(pred_files):
        raise ValueError("GT 與 PRED 檔案數量不一致！")

    metrics_list = []
    files = tqdm(zip(gt_files, pred_files), desc='metrics calculate', dynamic_ncols=True)
    for gt_file, pred_file in files:
        gt = load_depth(os.path.join(gt_dir, gt_file))
        pred = load_depth(os.path.join(pred_dir, pred_file))

        metrics = compute_metrics(gt, pred, align_way, cap_depth)
        metrics_list.append(metrics)

        # print(f"[Done] {gt_file} vs {pred_file}")

    metrics_arr = np.array(metrics_list)
    avg_metrics = metrics_arr.mean(axis=0)

    # 輸出結果
    with open(output_txt, "w") as f:
        f.write("Average Metrics (GT vs Pred)\n")
        f.write(f"RMSE: {avg_metrics[0]:.6f}\n")
        f.write(f"MAE: {avg_metrics[1]:.6f}\n")
        f.write(f"MRE: {avg_metrics[2]:.6f}\n")
        f.write(f"RMSElog: {avg_metrics[3]:.6f}\n")
        f.write(f"delta1: {avg_metrics[4]:.6f}\n")
        f.write(f"delta2: {avg_metrics[5]:.6f}\n")
        f.write(f"delta3: {avg_metrics[6]:.6f}\n")

    print(f"\n平均結果已存成 {output_txt}")


# --------------------------
# 使用範例
# --------------------------
if __name__ == "__main__":
    gt_dir = "dataset/test/gt/"
    pred_dir = "output_new/depthAnything_metric_raw_5_6_5_fold_padding_6/result_stitched_seamless/"  # prediction 資料夾
    output_txt = "output_new/depthAnything_metric_raw_5_6_5_fold_padding_6/metrics_result.txt"

    evaluate_folder(gt_dir, pred_dir, output_txt,
                    align_way=2, cap_depth=True)

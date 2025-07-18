import torch.nn.functional as F
import torch
import torch.nn as nn

def l1_loss(pred, gt, mask):
    return F.l1_loss(pred[mask], gt[mask])

def gradient_loss(pred, gt, mask):
    def gradient_x(img):
        return img[:, :, :, 1:] - img[:, :, :, :-1]

    def gradient_y(img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]

    # Crop mask to match gradient shape
    mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]

    pred_dx = gradient_x(pred)
    gt_dx = gradient_x(gt)
    pred_dy = gradient_y(pred)
    gt_dy = gradient_y(gt)

    loss_x = F.l1_loss(pred_dx[mask_x], gt_dx[mask_x])
    loss_y = F.l1_loss(pred_dy[mask_y], gt_dy[mask_y])
    return loss_x + loss_y

class Laplacian(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1)

def laplacian_loss(pred, gt, mask):
    lap_filter = Laplacian().to(pred.device)
    lap_pred = lap_filter(pred)
    lap_gt = lap_filter(gt)
    return F.l1_loss(lap_pred[mask], lap_gt[mask])

# def total_loss(pred, gt):
#     valid_mask = gt > 0
#     # pred = torch.clamp(pred, 0.1, 10.0)

#     loss = (
#         l1_loss(pred, gt, valid_mask)
#         + 0.5 * gradient_loss(pred, gt, valid_mask)
#         + 0.2 * laplacian_loss(pred, gt, valid_mask)
#     )
#     return loss

def total_loss(pred, gt, use_l1=True, use_grad=True, use_lap=True):
    valid_mask = gt > 0
    loss = 0.0

    if use_l1:
        loss += l1_loss(pred, gt, valid_mask)

    if use_grad:
        loss += 0.5 * gradient_loss(pred, gt, valid_mask)

    if use_lap:
        loss += 0.2 * laplacian_loss(pred, gt, valid_mask)

    return loss


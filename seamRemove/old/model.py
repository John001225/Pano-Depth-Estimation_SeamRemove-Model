import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.block(x)

def up(x): return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

class EfficientUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(EfficientUNet, self).__init__()

        # Load EfficientNet encoder with pretrained weights
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)
        self.encoder = backbone.features

        # Project 4-channel input (RGB + Depth) to 3 channels for EfficientNet
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),  # [B, 4, H, W] → [B, 3, H, W]
            nn.ReLU(inplace=True)
        )

        # Decoder layers with correct in/out channels
        self.up6 = UpBlock(320, 192)                # x7
        self.up5 = UpBlock(192 + 192, 112)          # d6 + x6
        self.up4 = UpBlock(112 + 112, 80)           # d5 + x5
        self.up3 = UpBlock(80 + 40, 40)             # d4 + x3
        self.up2 = UpBlock(40 + 24, 24)             # d3 + x2
        self.up1 = UpBlock(24 + 16, 16)             # d2 + x1

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)  # [B, 4, H, W] → [B, 3, H, W]

        # EfficientNet encoder
        x1 = self.encoder[0](x)         # [B, 16, H/2, W/2]
        x2 = self.encoder[1](x1)        # [B, 24, H/4, W/4]
        x3 = self.encoder[2](x2)        # [B, 40, H/8, W/8]
        x4 = self.encoder[3](x3)        # [B, 80, H/16, W/16]
        x5 = self.encoder[4](x4)        # [B, 112, H/16, W/16]
        x6 = self.encoder[5](x5)        # [B, 192, H/32, W/32]
        x7 = self.encoder[6](x6)        # [B, 320, H/32, W/32]
        x8 = self.encoder[7](x7)

        # print("x4", x4.shape)
        # print("x5", x5.shape)
        # print("x6", x6.shape)
        # print("x7", x7.shape)
        # print("x8", x8.shape)


        # Decoder
        d6 = self.up6(x8)                         # [B, 192, H/16, W/16]
        d5 = self.up5(torch.cat([d6, up(x7)], dim=1)) # [B, 112, H/16, W/16]
        d4 = self.up4(torch.cat([d5, up(x6)], dim=1)) # [B, 80, H/16, W/16]
        d3 = self.up3(torch.cat([d4, up(x4)], dim=1)) # [B, 40, H/8, W/8]
        d2 = self.up2(torch.cat([d3, up(x3)], dim=1)) # [B, 24, H/4, W/4]
        d1 = self.up1(torch.cat([d2, up(x2)], dim=1)) # [B, 16, H/2, W/2]

        # out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)  # → [B, 16, H, W]
        # out = self.final_conv(out)  # → [B, 1, H, W]
        out = self.final_conv(d1)  # [B, 1, H/2, W/2]
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)  # → [B, 1, H, W]

        return out


def get_model(in_channels=4, out_channels=1):
    return EfficientUNet(in_channels=in_channels, out_channels=out_channels)

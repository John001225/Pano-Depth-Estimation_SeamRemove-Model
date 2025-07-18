import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)  # Upsample

        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        return self.conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(EfficientUNet, self).__init__()

        # Load EfficientNet encoder with pretrained weights
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)
        self.encoder = backbone.features

        # Project 4-channel input (RGB + Depth) to 3 channels for EfficientNet
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Decoder layers
        self.up6 = UpBlock(320, 192)
        self.up5 = UpBlock(192, 112)
        self.up4 = UpBlock(112, 80)
        self.up3 = UpBlock(80, 40)
        self.up2 = UpBlock(40, 24)
        self.up1 = UpBlock(24, 16)

        # self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(24, out_channels, kernel_size=1)

    def forward(self, x):
        x_input = x  # Save for later upsampling
        x = self.input_proj(x)

        # Encoder
        x1 = self.encoder[0](x)         # [B, 16, H/2, W/2]
        x2 = self.encoder[1](x1)        # [B, 24, H/4, W/4]
        x3 = self.encoder[2](x2)        # [B, 40, H/8, W/8]
        x4 = self.encoder[3](x3)        # [B, 80, H/16, W/16]
        x5 = self.encoder[4](x4)        # [B, 112, H/16, W/16]
        x6 = self.encoder[5](x5)        # [B, 192, H/32, W/32]
        x7 = self.encoder[6](x6)        # [B, 320, H/32, W/32]
        x8 = self.encoder[7](x7)        # Same shape as x7

        # Decoder
        d6 = self.up6(x8, x7)  # [B, 192, H/16, W/16]
        d5 = self.up5(d6, x6)  # [B, 112, H/8, W/8]
        d4 = self.up4(d5, x5)  # [B, 80, H/8, W/8]
        d3 = self.up3(d4, x4)  # [B, 40, H/4, W/4]
        d2 = self.up2(d3, x3)  # [B, 24, H/2, W/2]
        # d1 = self.up1(d2, x2)  # [B, 16, H, W]

        out = self.final_conv(d2)  # [B, 1, H, W]
        return out


def get_model(in_channels=4, out_channels=1):
    return EfficientUNet(in_channels=in_channels, out_channels=out_channels)

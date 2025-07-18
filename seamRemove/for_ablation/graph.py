import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchviz import make_dot


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
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(EfficientUNet, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)
        self.encoder = backbone.features
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.up6 = UpBlock(320, 192)
        self.up5 = UpBlock(192, 112)
        self.up4 = UpBlock(112, 80)
        self.up3 = UpBlock(80, 40)
        self.up2 = UpBlock(40, 24)
        self.final_conv = nn.Conv2d(24, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)
        x6 = self.encoder[5](x5)
        x7 = self.encoder[6](x6)
        x8 = self.encoder[7](x7)
        d6 = self.up6(x8, x7)
        d5 = self.up5(d6, x6)
        d4 = self.up4(d5, x5)
        d3 = self.up3(d4, x4)
        d2 = self.up2(d3, x3)
        out = self.final_conv(d2)
        return out


model = EfficientUNet()
dummy_input = torch.randn(1, 4, 512, 512)
output = model(dummy_input)

# Generate architecture graph
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
output_path = "./efficientunet_512x512"
dot.render(output_path)

output_path + ".png"

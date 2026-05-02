"""U-Net model definition."""

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Small U-Net for single-channel ultrasound segmentation."""

    def __init__(self):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        self.enc1 = C(1, 64)
        self.enc2 = C(64, 128)
        self.enc3 = C(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(C(256, 512), nn.Dropout2d(0.3))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = C(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = C(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = C(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 input channel, got {x.shape[1]}")

        h, w = x.shape[2], x.shape[3]
        if h % 8 != 0 or w % 8 != 0:
            raise RuntimeError("Height and width must be divisible by 8")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            raise RuntimeError("Shape mismatch in skip connection (d3, e3)")
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            raise RuntimeError("Shape mismatch in skip connection (d2, e2)")
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            raise RuntimeError("Shape mismatch in skip connection (d1, e1)")
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))

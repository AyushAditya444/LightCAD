# lightcad_model.py
# Lightweight 3D U-Net ("LightCAD") for calcium-imaging denoising.
# - Spatial-only downsampling/upsampling (keeps time T unchanged)
# - Upsample to skip's exact size to avoid concat mismatches
# - GroupNorm groups chosen safely for any channel count

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd

def conv3x3(cin, cout, groups=1):
    return nn.Conv3d(cin, cout, kernel_size=3, padding=1, groups=groups, bias=False)

def make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Pick num_groups (<= max_groups) that divides num_channels; fallback to 1."""
    if num_channels <= 0:
        raise ValueError("num_channels must be > 0")
    g = gcd(num_channels, max_groups) or 1
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    if g < 1:
        g = 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)

class DWSeparable3D(nn.Module):
    """Depthwise-separable 3D conv block + residual + GroupNorm."""
    def __init__(self, c: int, expansion: int = 2, max_gn_groups: int = 8):
        super().__init__()
        self.dw  = conv3x3(c, c, groups=c)   # depthwise (per-channel)
        self.pw1 = nn.Conv3d(c, c * expansion, 1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pw2 = nn.Conv3d(c * expansion, c, 1, bias=False)
        self.gn  = make_gn(c, max_groups=max_gn_groups)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.act(x + y)        # residual
        return self.gn(y)

class Down(nn.Module):
    """
    Encoder down block:
      - Spatial-only pooling (kernel=(1,2,2), stride=(1,2,2)) keeps time T unchanged.
      - Pointwise conv + two DWSeparable3D blocks.
    """
    def __init__(self, cin: int, cout: int, max_gn_groups: int = 8):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, 1, bias=False),
            make_gn(cout, max_groups=max_gn_groups),
            nn.LeakyReLU(0.1, inplace=True),
            DWSeparable3D(cout, max_gn_groups=max_gn_groups),
            DWSeparable3D(cout, max_gn_groups=max_gn_groups),
        )

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    """
    Decoder up block:
      - Trilinear upsample to EXACT (T,H,W) of the skip
      - Concat along channels
      - Pointwise conv + two DWSeparable3D blocks
    """
    def __init__(self, cin: int, cout: int, max_gn_groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, 1, bias=False),
            make_gn(cout, max_groups=max_gn_groups),
            nn.LeakyReLU(0.1, inplace=True),
            DWSeparable3D(cout, max_gn_groups=max_gn_groups),
            DWSeparable3D(cout, max_gn_groups=max_gn_groups),
        )

    def forward(self, x, skip):
        # Upsample decoder feature to exact skip size (T,H,W)
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class LightCAD(nn.Module):
    """Compact U-Net-like denoiser. Base=12 fits 4 GB VRAM well."""
    def __init__(self, in_ch: int = 1, base: int = 12, max_gn_groups: int = 8, verbose: bool = True):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        if verbose:
            g1 = make_gn(c1, max_gn_groups).num_groups
            g2 = make_gn(c2, max_gn_groups).num_groups
            g3 = make_gn(c3, max_gn_groups).num_groups
            print(f"[LightCAD] channels: {c1},{c2},{c3} | GN groups: {g1},{g2},{g3}", flush=True)

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, c1, 3, padding=1, bias=False),
            make_gn(c1, max_groups=max_gn_groups),
            nn.LeakyReLU(0.1, inplace=True),
            DWSeparable3D(c1, max_gn_groups=max_gn_groups),
        )
        self.down1 = Down(c1, c2, max_gn_groups=max_gn_groups)  # T, H/2, W/2
        self.down2 = Down(c2, c3, max_gn_groups=max_gn_groups)  # T, H/4, W/4

        self.bot   = nn.Sequential(
            DWSeparable3D(c3, max_gn_groups=max_gn_groups),
            DWSeparable3D(c3, max_gn_groups=max_gn_groups),
        )

        self.up2   = Up(c3 + c3, c2, max_gn_groups=max_gn_groups)  # T, H/4, W/4
        self.up1   = Up(c2 + c2, c1, max_gn_groups=max_gn_groups)  # T, H/2, W/2

        self.head  = nn.Conv3d(c1, 1, 1)

    def forward(self, x):
        # x: [B,1,T,H,W]
        s1 = self.stem(x)    # [B,c1,T,H,W]
        s2 = self.down1(s1)  # [B,c2,T,H/2,W/2]
        s3 = self.down2(s2)  # [B,c3,T,H/4,W/4]
        b  = self.bot(s3)    # [B,c3,T,H/4,W/4]
        u2 = self.up2(b, s3) # [B,c2,T,H/4,W/4]
        u1 = self.up1(u2, s2)# [B,c1,T,H/2,W/2]
        out = self.head(u1)                           # [B,1,T,H/2,W/2]
        out = F.interpolate(out, size=x.shape[2:],    # upsample to input spatial size
                            mode="trilinear", align_corners=False)
        return out
        return out

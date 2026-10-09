"""The Mola network: a 1D U-Net denoiser conditioned on DNA sequence."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # sinusoidal embedding, as in openai/glide-text2im
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, dilation=1, groups=1):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=17, padding=8 * dilation, dilation=dilation, bias=False, groups=groups),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=17, padding=8 * dilation, dilation=dilation, bias=False, groups=groups),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, groups=1, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, in_channels, residual=True, groups=groups),
            DoubleConv(in_channels, in_channels, residual=True, groups=groups),
            DoubleConv(in_channels, out_channels, groups=groups),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        return x + self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, groups=1, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, groups=groups),
            DoubleConv(in_channels, in_channels, residual=True, groups=groups),
            DoubleConv(in_channels, out_channels, in_channels // 2, groups=groups),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = self.conv(torch.cat([skip_x, x], dim=1))
        return x + self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])


class UNet(nn.Module):
    def __init__(self, time_dim=256, seq_channels=4):
        super().__init__()
        self.inc = DoubleConv(2 + seq_channels, 512)
        # 10 kb -> 25 bins of 400 bp
        self.down1 = Down(512, 512, groups=1, scale_factor=4)
        self.down2 = Down(512, 512, groups=2, scale_factor=4)
        self.down3 = Down(512, 512, groups=4, scale_factor=5)
        self.down4 = Down(512, 512, groups=8, scale_factor=5)

        self.bot1 = DoubleConv(512, 512, groups=32)
        self.bot2 = DoubleConv(512, 512, groups=32)
        self.bot3 = DoubleConv(512, 512, groups=32)

        self.up1 = Up(1024, 512, groups=8, scale_factor=5)
        self.up2 = Up(1024, 512, groups=4, scale_factor=5)
        self.up3 = Up(1024, 512, groups=2, scale_factor=4)
        self.up4 = Up(1024, 512, groups=1, scale_factor=4)
        self.outc = nn.Conv1d(512, 2, kernel_size=1)

        self.time_embedding = TimestepEmbedder(time_dim)

    def forward(self, x, seq, t):
        x = torch.cat([x, seq], dim=1)
        t = self.time_embedding(t)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)

        x5 = self.bot3(self.bot2(self.bot1(x5)))

        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        return self.outc(x)


class Mola(nn.Module):
    def __init__(self, time_dim=256):
        super().__init__()
        self.unet = UNet(time_dim=time_dim, seq_channels=4)

    def forward(self, x, seq, t):
        return self.unet(x, seq, t)


def load_weights(path, device="cpu"):
    state_dict = torch.load(path, map_location=device)
    # checkpoints saved from nn.DataParallel have a "module." prefix
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model(path="resources/mola.pth", device="cuda"):
    """Load a trained Mola checkpoint for inference."""
    model = Mola()
    model.load_state_dict(load_weights(path, device))
    return model.to(device).eval()

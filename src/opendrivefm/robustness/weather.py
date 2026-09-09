"""Weather and sensor degradation with a physical basis.

The rain in perturbations.py paints opaque vertical bars two pixels wide. That
is not what rain does to a camera and it is not a fair test: a hard edge is
trivially detectable by a variance-based trust head, so the model looks better
at catching rain than it would be in the field.

These are written from how each effect actually forms on the sensor:

  rain   streaks are slanted by the vehicle's motion, thin, semi-transparent
         and motion-blurred ALONG their own axis; the background behind heavy
         rain also loses contrast to veiling from out-of-focus near drops.
  fog    atmospheric scattering, I = I0*t + A*(1-t) with transmission
         t = exp(-beta*d). No depth is available in a 90x160 input tensor, so
         d is approximated by image row -- the standard cheap proxy, and the
         reason fog here is a lower bound on the real effect.
  snow   bright, defocused blobs of varying radius, not single pixels. Snow
         occludes in patches; salt-and-pepper noise does not.
"""
from __future__ import annotations
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F


def _blur(x, sigma):
    k = max(3, 2 * int(math.ceil(2 * sigma)) + 1)
    c = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(C, 1, 1, k), padding=(0, k // 2), groups=C)
    return F.conv2d(x, g.view(1, 1, k, 1).expand(C, 1, k, 1), padding=(k // 2, 0), groups=C)


class RainStreaksV2(nn.Module):
    """Slanted, motion-blurred, semi-transparent streaks plus veiling."""
    def __init__(self, n=(140, 260), alpha=(0.30, 0.62), slant_deg=(8.0, 22.0),
                 length=(0.10, 0.30), veil=(0.05, 0.12)):
        super().__init__()
        self.n, self.alpha, self.slant, self.length, self.veil = n, alpha, slant_deg, length, veil

    def forward(self, x):
        B, C, H, W = x.shape
        streak = torch.zeros(1, 1, H, W, device=x.device)
        n = random.randint(*self.n)
        th = math.radians(random.uniform(*self.slant))
        dx, dy = math.sin(th), math.cos(th)
        for _ in range(n):
            L = int(random.uniform(*self.length) * H)
            x0 = random.uniform(0, W); y0 = random.uniform(-0.1 * H, H)
            a = random.uniform(*self.alpha)
            for s in range(max(2, L)):
                xi, yi = int(x0 + dx * s), int(y0 + dy * s)
                if 0 <= xi < W and 0 <= yi < H:
                    streak[0, 0, yi, xi] = max(float(streak[0, 0, yi, xi]), a)
        # motion blur ALONG the streak, which is what makes a drop a streak
        streak = _blur(streak, 0.7)
        out = x * (1 - streak) + streak * 0.92                     # drops are bright
        v = random.uniform(*self.veil)
        return (out * (1 - v) + v * 0.75).clamp(0, 1)              # veiling contrast loss


class Fog(nn.Module):
    """I = I0*t + A*(1-t), t = exp(-beta*d), d approximated by image row."""
    def __init__(self, beta=(1.1, 2.4), airlight=(0.72, 0.90)):
        super().__init__()
        self.beta, self.airlight = beta, airlight

    def forward(self, x):
        B, C, H, W = x.shape
        beta = random.uniform(*self.beta); A = random.uniform(*self.airlight)
        row = torch.linspace(1.0, 0.0, H, device=x.device).view(1, 1, H, 1)
        d = 0.25 + 0.75 * row                      # near at the bottom, far at the top
        t = torch.exp(-beta * d)
        return (x * t + A * (1 - t)).clamp(0, 1)


class Snow(nn.Module):
    """Bright defocused blobs. Occludes in patches, unlike per-pixel noise."""
    def __init__(self, n=(90, 190), radius=(0.6, 2.4), alpha=(0.55, 0.95), veil=(0.03, 0.10)):
        super().__init__()
        self.n, self.radius, self.alpha, self.veil = n, radius, alpha, veil

    def forward(self, x):
        B, C, H, W = x.shape
        flake = torch.zeros(1, 1, H, W, device=x.device)
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                                torch.arange(W, device=x.device, dtype=torch.float32),
                                indexing="ij")
        for _ in range(random.randint(*self.n)):
            r = random.uniform(*self.radius); a = random.uniform(*self.alpha)
            cx, cy = random.uniform(0, W), random.uniform(0, H)
            flake = torch.maximum(flake, a * torch.exp(
                -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))).view(1, 1, H, W))
        flake = _blur(flake, 0.5)
        out = x * (1 - flake) + flake * 0.97
        v = random.uniform(*self.veil)
        return (out * (1 - v) + v * 0.85).clamp(0, 1)


WEATHER = {"rain": RainStreaksV2, "fog": Fog, "snow": Snow}

#UnetGenerator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    """
    Standard U-Net generator producing single-channel perturbation.
    Input: (B,1,H,W), Output: (B,1,H,W)
    """
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1), nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters*8, base_filters*8, 3, padding=1), nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters*16, base_filters*16, 3, padding=1), nn.ReLU()
        )
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_filters*16, base_filters*8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters*8, base_filters*8, 3, padding=1), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, padding=1), nn.ReLU()
        )
        # Perturbation head
        self.final = nn.Conv2d(base_filters, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        # Bottleneck
        b = self.bottleneck(p4)
        # Decoder
        u4 = self.up4(b)
        c4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(c4)
        u3 = self.up3(d4)
        c3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(c3)
        u2 = self.up2(d3)
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(c2)
        u1 = self.up1(d2)
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(c1)
        # Head
        out = self.tanh(self.final(d1))
        return out

class AdversarialFFTUNet(nn.Module):
    """
    Full model combining FFT-based frequency block, U-Net generator, and reconstruction.
    """
    def __init__(self, cutoff_radius:float=10.0):
        super().__init__()
        self.cutoff_radius = nn.Parameter(torch.tensor(cutoff_radius))
        self.unet = UNetGenerator(in_channels=1)

    def forward(self, rgb: torch.Tensor):
        # rgb: (B,3,H,W), values in [0,1]
        R, G, B = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
        Y  = 0.299*R + 0.587*G + 0.114*B
        Cr = 0.5*R - 0.4187*G - 0.0813*B
        Cb = -0.1687*R - 0.3313*G + 0.5*B

        # FFT -> mask -> IFFT on Y
        spec = torch.fft.fft2(Y, norm='ortho')
        spec_shifted = torch.fft.fftshift(spec)

        device = spec_shifted.device

        H = spec_shifted.shape[2]
        W = spec_shifted.shape[3]

        mask = torch.ones(H, W, dtype=torch.float32, device=device)
        center_h, center_w = H//2, W//2
        y = torch.arange(H, device=device).reshape(-1, 1)
        x = torch.arange(W, device=device).reshape(1, -1)
        distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2) 

        mask[distance <= self.cutoff_radius] = 0

        spec_mask = spec_shifted * mask

        spec_unshifted = torch.fft.ifftshift(spec_mask)
        Yhf = torch.fft.ifft2(spec_unshifted, norm='ortho').real
        # UNet perturbation on high-freq Y
        deltaY = self.unet(Yhf)
        # Add perturbation
        Y_adv = torch.clamp(Y + deltaY, 0.0, 1.0)
        # Reconstruct RGB
        R_adv = Y_adv + 1.402*Cr
        G_adv = Y_adv - 0.3441*Cb - 0.7141*Cr
        B_adv = Y_adv + 1.772*Cb
        rgb_adv = torch.cat([R_adv, G_adv, B_adv], dim=1)
        return rgb_adv, deltaY

# Example instantiation
# model = AdversarialFFTUNet(cutoff_radius=10)

import os
import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import torch.nn.functional as F
import lpips
import clip
import torchvision.models as models
import torchvision

# ────────────────────────────────────────────────────────────────────────────────
# 1. MODEL DEFINITION (single-image support)
# ────────────────────────────────────────────────────────────────────────────────

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
    def __init__(self, cutoff_radius: float = 10.0):
        super().__init__()
        self.cutoff_radius = nn.Parameter(torch.tensor(cutoff_radius))
        self.unet = UNetGenerator(in_channels=1)

    def forward(self, rgb: torch.Tensor):
        # Standard forward returning only rgb_adv
        rgb_adv = self._process(rgb)
        return rgb_adv

    def forward_with_intermediates(self, rgb: torch.Tensor):
        # Ensure batch dim
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        # Split channels
        R, G, B = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
        # YCbCr
        Y  = 0.299*R + 0.587*G + 0.114*B
        Cr = 0.5*R - 0.4187*G - 0.0813*B
        Cb = -0.1687*R - 0.3313*G + 0.5*B
        # FFT + shift
        spec = torch.fft.fft2(Y, norm='ortho')
        spec_shifted = torch.fft.fftshift(spec)
        # Mask
        Bn,_,H,W = spec_shifted.shape
        device = spec_shifted.device
        y = torch.arange(H, device=device).view(-1,1)
        x = torch.arange(W, device=device).view(1,-1)
        dist = torch.sqrt((y - H//2)**2 + (x - W//2)**2)
        mask = (dist > self.cutoff_radius).float()
        spec_mask = spec_shifted * mask
        # IFFT
        spec_unshifted = torch.fft.ifftshift(spec_mask)
        Yhf = torch.fft.ifft2(spec_unshifted, norm='ortho').real
        # U-Net perturbation
        deltaY = self.unet(Yhf)
        Y_adv = torch.clamp(Y + deltaY, 0.0, 1.0)
        # Reconstruct RGB
        R_adv = Y_adv + 1.402*Cr
        G_adv = Y_adv - 0.3441*Cb - 0.7141*Cr
        B_adv = Y_adv + 1.772*Cb
        rgb_adv = torch.cat([R_adv, G_adv, B_adv], dim=1)
        # Remove batch dim
        rgb_adv = rgb_adv.squeeze(0)
        # Return all intermediates
        return rgb_adv, (Y.squeeze(0), spec_shifted.squeeze(0), mask, spec_mask.squeeze(0), Yhf.squeeze(0), deltaY.squeeze(0), Y_adv.squeeze(0))


def pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    _,_,H,W = x.shape
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    return F.pad(x, (0,pw,0,ph), mode='reflect'), H, W

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & SETUP
# ────────────────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "./checkpoints/model_epoch003_batch0200.pt"
INPUT_PATH = "./input/img2.jpg"
OUTPUT_DIR = "./outputs_single"
os.makedirs(OUTPUT_DIR, exist_ok=True)

to_tensor = transforms.ToTensor()
resnet_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])
clip_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073],
                         std=[0.26862954,0.26130258,0.27577711]),
])
lpips_fn   = lpips.LPIPS(net='alex').to(DEVICE)
resnet50   = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
clip_model,_ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()

model = AdversarialFFTUNet(cutoff_radius=10.0).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
else:
    model.load_state_dict(ckpt)
model.eval()

# ────────────────────────────────────────────────────────────────────────────────
# RUN INFERENCE WITH INTERMEDIATES
# ────────────────────────────────────────────────────────────────────────────────
img = Image.open(INPUT_PATH).convert("RGB")
tensor = to_tensor(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
padded, orig_H, orig_W = pad_to_multiple(tensor)

with torch.no_grad():
    perturbed, (Y, spec_shifted, mask, spec_mask, Yhf, deltaY, Y_adv) = model.forward_with_intermediates(padded)

# Crop back to original size
perturbed = perturbed[:, :orig_H, :orig_W]

# ────────────────────────────────────────────────────────────────────────────────
# EVALUATE & PRINT METRICS
# ────────────────────────────────────────────────────────────────────────────────
# Compute LPIPS
orig_lp = tensor * 2 - 1
adv_lp  = perturbed.unsqueeze(0) * 2 - 1
lpips_val = lpips_fn(orig_lp, adv_lp).item()

# ResNet50 inference
orig_res = resnet_tf(tensor)
adv_res  = resnet_tf(perturbed.unsqueeze(0))
with torch.no_grad():
    logits_o = resnet50(orig_res)
    logits_a = resnet50(adv_res)
    prob_o   = torch.softmax(logits_o, dim=1)
    prob_a   = torch.softmax(logits_a, dim=1)
    label_o  = prob_o.argmax(1).item()
    label_a  = prob_a.argmax(1).item()
    conf_o   = prob_o[0, label_o].item()
    conf_a   = prob_a[0, label_a].item()

# CLIP inference
orig_clip = clip_tf(tensor)
adv_clip  = clip_tf(perturbed.unsqueeze(0))
with torch.no_grad():
    emb_o = clip_model.encode_image(orig_clip)
    emb_a = clip_model.encode_image(adv_clip)
    emb_o /= emb_o.norm(dim=-1, keepdim=True)
    emb_a /= emb_a.norm(dim=-1, keepdim=True)
    cos_sim = F.cosine_similarity(emb_o, emb_a).item()

# Print all metrics
print(f"LPIPS distance:           {lpips_val:.4f}")
print(f"ResNet50 orig label:      {label_o}, conf: {conf_o:.4f}")
print(f"ResNet50 adv  label:      {label_a}, conf: {conf_a:.4f}")
print(f"ResNet50 fooled:          {label_o != label_a}")
print(f"CLIP cosine similarity:   {cos_sim:.4f}")

# ────────────────────────────────────────────────────────────────────────────────
# SAVE RGB & INTERMEDIATE IMAGES
# ────────────────────────────────────────────────────────────────────────────────
# RGB
utils.save_image(tensor,    os.path.join(OUTPUT_DIR, "original.png"))
utils.save_image(perturbed.unsqueeze(0), os.path.join(OUTPUT_DIR, "perturbed.png"))

# Intermediates (normalized for visualization)
utils.save_image(Y,           os.path.join(OUTPUT_DIR, "Y.png"),             normalize=True)
utils.save_image(spec_shifted.abs(), os.path.join(OUTPUT_DIR, "spec_shifted.png"), normalize=True)
utils.save_image(mask.unsqueeze(0),  os.path.join(OUTPUT_DIR, "mask.png"),          normalize=True)
utils.save_image(spec_mask.abs(),    os.path.join(OUTPUT_DIR, "spec_mask.png"),     normalize=True)
utils.save_image(Yhf,         os.path.join(OUTPUT_DIR, "Yhf.png"),           normalize=True)
utils.save_image(deltaY,      os.path.join(OUTPUT_DIR, "deltaY.png"),        normalize=True)
utils.save_image(Y_adv,       os.path.join(OUTPUT_DIR, "Y_adv.png"),         normalize=True)

print(f"All outputs saved to {OUTPUT_DIR}")
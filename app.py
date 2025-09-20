import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from PIL import Image
import lpips
import clip
import torchvision.models as models
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import streamlit as st

# ────────────────────────────────────────────────────────────────
# 1. MODEL DEFINITION
# ────────────────────────────────────────────────────────────────
class UNetGenerator(nn.Module):
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
    
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        u4 = self.up4(b); c4 = torch.cat([u4, e4], dim=1); d4 = self.dec4(c4)
        u3 = self.up3(d4); c3 = torch.cat([u3, e3], dim=1); d3 = self.dec3(c3)
        u2 = self.up2(d3); c2 = torch.cat([u2, e2], dim=1); d2 = self.dec2(c2)
        u1 = self.up1(d2); c1 = torch.cat([u1, e1], dim=1); d1 = self.dec1(c1)
        return self.tanh(self.final(d1))

class AdversarialFFTUNet(nn.Module):
    def __init__(self, cutoff_radius=0.0):
        super().__init__()
        self.cutoff_radius = nn.Parameter(torch.tensor(cutoff_radius))
        self.unet = UNetGenerator(in_channels=1)

    def forward_with_intermediates(self, rgb):
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        R, G, B = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
        Y  = 0.299*R + 0.587*G + 0.114*B
        Cr = 0.5*R - 0.4187*G - 0.0813*B
        Cb = -0.1687*R - 0.3313*G + 0.5*B
        spec = torch.fft.fft2(Y, norm='ortho')
        spec_shifted = torch.fft.fftshift(spec)
        Bn,_,H,W = spec_shifted.shape
        device = spec_shifted.device
        y = torch.arange(H, device=device).view(-1,1)
        x = torch.arange(W, device=device).view(1,-1)
        dist = torch.sqrt((y - H//2)**2 + (x - W//2)**2)
        mask = (dist > self.cutoff_radius).float()
        spec_mask = spec_shifted * mask
        spec_unshifted = torch.fft.ifftshift(spec_mask)
        Yhf = torch.fft.ifft2(spec_unshifted, norm='ortho').real
        deltaY = self.unet(Yhf)
        Y_adv = torch.clamp(Y + deltaY, 0.0, 1.0)
        R_adv = Y_adv + 1.402*Cr
        G_adv = Y_adv - 0.3441*Cb - 0.7141*Cr
        B_adv = Y_adv + 1.772*Cb
        rgb_adv = torch.cat([R_adv, G_adv, B_adv], dim=1).squeeze(0)
        return rgb_adv, (Y.squeeze(0), spec_shifted.squeeze(0), mask, spec_mask.squeeze(0), Yhf.squeeze(0), deltaY.squeeze(0), Y_adv.squeeze(0))

def pad_to_multiple(x, multiple=16):
    _,_,H,W = x.shape
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    return F.pad(x, (0,pw,0,ph), mode='reflect'), H, W

# ────────────────────────────────────────────────────────────────
# 2. SETUP
# ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "./development/Trained_models/checkpoints/model_epoch009_batch0200.pt"

to_tensor = transforms.ToTensor()
resnet_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
clip_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]),
])

lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
clip_model,_ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()

model = AdversarialFFTUNet(cutoff_radius=10.0).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
else:
    model.load_state_dict(ckpt)
model.eval()

# ────────────────────────────────────────────────────────────────
# 3. STREAMLIT APP
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPECIOUS Demo", layout="wide")
st.title("SPECIOUS: Spectral Perturbation Engine for Contrastive Inference Over Universal Surrogates [[Paper]](https://snookydru.github.io/SPECIOUS/) [[GitHub]](https://github.com/SnookyDru/SPECIOUS)")
st.caption("Developed by: Dhruv Kumar [[Github]](https://github.com/SnookyDru), Harshveer Singh [[Github]](https://github.com/Harshveer03), Deepanshu Singh [[Github]](https://github.com/deepanshusingh963n)")
st.caption("Affiliation: USAR, Guru Gobind Singh Indraprastha University, Delhi, India")
st.markdown("Upload an image to generate its adversarial counterpart using the SPECIOUS model. Visualize intermediate steps and evaluate the perturbation using various metrics.")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
    padded, orig_H, orig_W = pad_to_multiple(tensor)

    with torch.no_grad():
        perturbed, (Y, spec_shifted, mask, spec_mask, Yhf, deltaY, Y_adv) = model.forward_with_intermediates(padded)

    perturbed = perturbed[:, :orig_H, :orig_W]

    # ───────────────────────────────────────────
    # Metrics
    # ───────────────────────────────────────────
    orig_lp = tensor * 2 - 1
    adv_lp  = perturbed.unsqueeze(0) * 2 - 1
    lpips_val = lpips_fn(orig_lp, adv_lp).item()

    ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_val = ssim(tensor, perturbed.unsqueeze(0)).item()
    psnr_val = psnr(tensor, perturbed.unsqueeze(0)).item()

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

    orig_clip = clip_tf(tensor)
    adv_clip  = clip_tf(perturbed.unsqueeze(0))
    with torch.no_grad():
        emb_o = clip_model.encode_image(orig_clip)
        emb_a = clip_model.encode_image(adv_clip)
        emb_o /= emb_o.norm(dim=-1, keepdim=True)
        emb_a /= emb_a.norm(dim=-1, keepdim=True)
        cos_sim = F.cosine_similarity(emb_o, emb_a).item()

    # ───────────────────────────────────────────
    # Helper for safe display
    # ───────────────────────────────────────────
    def to_display(tensor):
        if tensor.dim() == 3:
            img = tensor.permute(1, 2, 0).detach().cpu().numpy()
        else:
            img = tensor.detach().cpu().numpy()
        return img.clip(0, 1)

    # ───────────────────────────────────────────
    # Show Metrics
    # ───────────────────────────────────────────
    st.subheader("📊 Evaluation Metrics")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("LPIPS Distance", f"{lpips_val:.4f}")
        st.metric("SSIM", f"{ssim_val:.4f}")
    with m2:
        st.metric("PSNR", f"{psnr_val:.2f} dB")
        st.metric("CLIP Cosine Similarity", f"{cos_sim:.4f}")
    with m3:
        st.metric("ResNet50 Orig Label", f"{label_o} (Conf: {conf_o *100:.2f}%)")
        st.metric("ResNet50 Adv Label", f"{label_a} (Conf: {conf_a *100:.2f}%)")

    st.markdown(f"**ResNet50 Fooled:** `{label_o != label_a}`")

    # ───────────────────────────────────────────
    # Show Images
    # ───────────────────────────────────────────
    st.subheader("🖼️ Original vs Perturbed Image")
    col1, col2 = st.columns(2)
    with col1:
        st.image(to_display(tensor.squeeze(0)), caption="Original Image", use_container_width=True)
    with col2:
        st.image(to_display(perturbed), caption="Perturbed Image", use_container_width=True)

    st.subheader("🔎 Intermediate Visualizations")
    cols = st.columns(3)
    with cols[0]:
        st.image(to_display(Y), caption="Luminance (Y)", use_container_width=True)
        st.image(to_display(spec_shifted.abs()), caption="FFT Shifted Magnitude", use_container_width=True)
    with cols[1]:
        st.image(to_display(mask.unsqueeze(0)), caption="Frequency Mask", use_container_width=True)
        st.image(to_display(spec_mask.abs()), caption="Masked Spectrum", use_container_width=True)
    with cols[2]:
        st.image(to_display(Yhf), caption="High-Frequency Y (Yhf)", use_container_width=True)
        st.image(to_display(deltaY), caption="Perturbation (ΔY)", use_container_width=True)
        st.image(to_display(Y_adv), caption="Adversarial Y", use_container_width=True)

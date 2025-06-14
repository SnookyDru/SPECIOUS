# test_fooling_clip_dynamic_cifar100.py

import os
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import lpips
import clip
from torchvision.datasets import CIFAR100

from dataset import SpeciousDataset
from UNetGenerator import AdversarialFFTUNet

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH  = "./checkpoints/model_epoch003_batch0200.pt"
TEST_IMG_DIR     = "./test_images"
BATCH_SIZE       = 1
NUM_WORKERS      = 0

OUTPUT_DIR       = Path("clip_fooling_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_PATH         = OUTPUT_DIR / "results.csv"

LPIPS_THRESHOLD     = 0.015    # perceptual threshold
ZS_CHANGE_THRESHOLD = 0.0    # zero-shot “fooled” if top-1 label changes

# ────────────────────────────────────────────────────────────────────────────────
# LOAD MODELS & TRANSFORMS
# ────────────────────────────────────────────────────────────────────────────────
# 1) LPIPS perceptual metric
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

# 2) CLIP zero-shot: model + prompts for CIFAR-100
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# CIFAR-100 classes & tokenized text prompts
dataset_cifar = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
text_prompts = clip.tokenize([f"a photo of a {c}" for c in dataset_cifar.classes]).to(DEVICE)
with torch.no_grad():
    text_features = clip_model.encode_text(text_prompts)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# CLIP preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

# 3) Adversarial generator
adv_model = AdversarialFFTUNet(cutoff_radius=10.0).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
state = ckpt.get("model_state", ckpt)
adv_model.load_state_dict(state)
adv_model.eval()

# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ────────────────────────────────────────────────────────────────────────────────
test_ds     = SpeciousDataset(TEST_IMG_DIR)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ────────────────────────────────────────────────────────────────────────────────
# RUN EVALUATION
# ────────────────────────────────────────────────────────────────────────────────
results = []
total   = fooled_lpips = fooled_zs = 0
sum_zs_drop_all = sum_zs_drop_fooled = 0.0
count_zs_fooled = 0

with torch.no_grad():
    for imgs in test_loader:
        imgs     = imgs.to(DEVICE)
        adv_imgs, _ = adv_model(imgs)
        
        # LPIPS
        lpips_val = lpips_fn(imgs*2-1, adv_imgs*2-1).item()
        lpips_fooled = int(lpips_val > LPIPS_THRESHOLD)
        fooled_lpips += lpips_fooled

        # CLIP zero-shot
        orig_in = preprocess(imgs)
        adv_in  = preprocess(adv_imgs)

        orig_emb = clip_model.encode_image(orig_in)
        adv_emb  = clip_model.encode_image(adv_in)
        orig_emb = orig_emb / orig_emb.norm(dim=-1, keepdim=True)
        adv_emb  = adv_emb  / adv_emb.norm(dim=-1, keepdim=True)

        # logits & probs
        orig_logits = (orig_emb @ text_features.T) * 100.0
        adv_logits  = (adv_emb  @ text_features.T) * 100.0
        orig_probs  = orig_logits.softmax(dim=-1)
        adv_probs   = adv_logits.softmax(dim=-1)

        # Top-1 predictions & confidences
        orig_idx  = orig_probs.argmax(dim=-1).item()
        adv_idx   = adv_probs.argmax(dim=-1).item()
        orig_conf = orig_probs[0, orig_idx].item()
        adv_conf  = adv_probs[0, orig_idx].item()

        zs_drop   = orig_conf - adv_conf
        zs_fooled = int(orig_idx != adv_idx)

        fooled_zs       += zs_fooled
        sum_zs_drop_all += zs_drop
        if zs_fooled:
            sum_zs_drop_fooled += zs_drop
            count_zs_fooled     += 1

        # Print per-image metrics
        print(f"Image {total} metrics:")
        print(f"  LPIPS: {lpips_val:.4f} (fooled: {bool(lpips_fooled)})")
        print(f"  Orig ZS label: {dataset_cifar.classes[orig_idx]} ({orig_conf:.4f})")
        print(f"  Adv ZS label:  {dataset_cifar.classes[adv_idx]} ({adv_probs[0, adv_idx]:.4f})")
        print(f"  Adv conf of orig label: {adv_conf:.4f}")
        print(f"  Zero-shot drop: {zs_drop:.4f} (fooled: {bool(zs_fooled)})\n")

        results.append({
            "index":                total,
            "lpips":                lpips_val,
            "lpips_fooled":         lpips_fooled,
            "orig_zs_idx":          orig_idx,
            "orig_zs_label":        dataset_cifar.classes[orig_idx],
            "orig_zs_conf":         orig_conf,
            "adv_zs_idx":           adv_idx,
            "adv_zs_label":         dataset_cifar.classes[adv_idx],
            "adv_zs_conf_of_orig":  adv_conf,
            "zs_confidence_drop":   zs_drop,
            "zs_fooled":            zs_fooled,
        })

        total += 1

# Final summaries
lpips_rate = fooled_lpips / total * 100.0
zs_rate    = fooled_zs      / total * 100.0
avg_zs_drop_all   = sum_zs_drop_all   / total         if total else 0.0
avg_zs_drop_fooled= sum_zs_drop_fooled/ count_zs_fooled if count_zs_fooled else 0.0

print(f"Total images:                    {total}")
print(f"LPIPS fooling rate:              {lpips_rate:.2f}%")
print(f"Zero-shot CLIP fooling rate:     {zs_rate:.2f}%")
print(f"Avg zero-shot drop (all):        {avg_zs_drop_all:.4f}")
print(f"Avg zero-shot drop (fooled):     {avg_zs_drop_fooled:.4f}")

# Save CSV (using explicit file handle to avoid permission issues)
try:
    with open(CSV_PATH, 'w', newline='') as f:
        pd.DataFrame(results).to_csv(f, index=False)
    print(f"Results successfully saved to {CSV_PATH}")
except PermissionError:
    print(f"Permission denied: could not write to {CSV_PATH}.")
    alt_path = OUTPUT_DIR / "results_safe.csv"
    with open(alt_path, 'w', newline='') as f:
        pd.DataFrame(results).to_csv(f, index=False)
    print(f"Saved to alternative path: {alt_path}")

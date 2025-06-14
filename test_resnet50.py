import os
from pathlib import Path
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.functional import softmax
import lpips

from dataset import SpeciousDataset
from UNetGenerator import AdversarialFFTUNet

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/model_epoch003_batch0200.pt"
TEST_IMG_DIR    = "./test_images"
BATCH_SIZE      = 1
NUM_WORKERS     = 0

OUTPUT_DIR      = Path("resnet50_fooling_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_PATH        = OUTPUT_DIR / "results.csv"

# ────────────────────────────────────────────────────────────────────────────────
# LOAD IMAGENET LABELS
# ────────────────────────────────────────────────────────────────────────────────
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
labels_path = OUTPUT_DIR / "imagenet_class_index.json"
if not labels_path.exists():
    import urllib.request
    urllib.request.urlretrieve(LABELS_URL, labels_path)

with open(labels_path) as f:
    class_idx = json.load(f)
idx2label = {int(k): v[1] for k, v in class_idx.items()}

# ────────────────────────────────────────────────────────────────────────────────
# MODELS & TRANSFORMS
# ────────────────────────────────────────────────────────────────────────────────
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
resnet50.eval()

imagenet_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

adv_model = AdversarialFFTUNet(cutoff_radius=10.0).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    adv_model.load_state_dict(ckpt["model_state"])
else:
    adv_model.load_state_dict(ckpt)
adv_model.eval()

# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ────────────────────────────────────────────────────────────────────────────────
test_ds = SpeciousDataset(TEST_IMG_DIR)
test_loader = DataLoader(test_ds,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS,
                         pin_memory=True)

# ────────────────────────────────────────────────────────────────────────────────
# RUN & EVALUATE
# ────────────────────────────────────────────────────────────────────────────────
results = []
total  = 0
fooled = 0
sum_conf_drop_all    = 0.0
sum_conf_drop_fooled = 0.0
count_fooled         = 0

with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(DEVICE)
        adv_imgs, _ = adv_model(imgs)

        # LPIPS
        orig_lp = imgs * 2 - 1
        adv_lp  = adv_imgs * 2 - 1
        lpips_val = lpips_fn(orig_lp, adv_lp).item()

        # ResNet prep & inference
        orig_in = imagenet_tf(imgs)
        adv_in  = imagenet_tf(adv_imgs)
        orig_probs = softmax(resnet50(orig_in), dim=1)
        adv_probs  = softmax(resnet50(adv_in), dim=1)

        orig_idx      = orig_probs.argmax(dim=1).item()
        adv_idx       = adv_probs.argmax(dim=1).item()
        orig_conf     = orig_probs[0, orig_idx].item()
        # New: confidence of original label on adversarial image
        adv_orig_conf = adv_probs[0, orig_idx].item()
        # Confidence drop is based on original label's drop
        conf_drop     = orig_conf - adv_orig_conf

        is_fooled     = (orig_idx != adv_idx)

        # Print
        print(f"Image {total}: LPIPS={lpips_val:.4f}  "
              f"Orig=[{orig_idx}] {idx2label[orig_idx]}({orig_conf:.4f})  "
              f"AdvPred=[{adv_idx}] {idx2label[adv_idx]}({adv_probs[0, adv_idx]:.4f})  "
              f"AdvOrigLabelConf={adv_orig_conf:.4f}  "
              f"Drop={conf_drop:.4f}  Fooled={is_fooled}")

        # Record
        results.append({
            "index":               total,
            "lpips":               lpips_val,
            "orig_idx":            orig_idx,
            "orig_label":          idx2label[orig_idx],
            "orig_confidence":     orig_conf,
            "adv_idx":             adv_idx,
            "adv_label":           idx2label[adv_idx],
            "adv_confidence":      adv_probs[0, adv_idx].item(),
            "adv_orig_confidence": adv_orig_conf,
            "confidence_drop":     conf_drop,
            "fooled":              int(is_fooled),
        })

        # Bookkeeping
        total += 1
        sum_conf_drop_all += conf_drop
        if is_fooled:
            fooled += 1
            sum_conf_drop_fooled += conf_drop
            count_fooled += 1

# Summary
fooling_rate = fooled / total * 100.0 if total > 0 else 0.0
avg_drop_all    = sum_conf_drop_all    / total       if total > 0       else 0.0
avg_drop_fooled = sum_conf_drop_fooled / count_fooled if count_fooled > 0 else 0.0

print(f"\nTotal images:               {total}")
print(f"Fooled:                     {fooled}")
print(f"Fooling rate:               {fooling_rate:.2f}%")
print(f"Avg confidence drop (all):  {avg_drop_all:.4f}")
print(f"Avg confidence drop (fooled): {avg_drop_fooled:.4f}")

# Save to CSV
pd.DataFrame(results).to_csv(CSV_PATH, index=False)
print(f"\nResults saved to {CSV_PATH}")

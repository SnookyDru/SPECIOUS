# train.py

import os
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import SpeciousDataset
from UNetGenerator import AdversarialFFTUNet
from speciousLoss import SpeciousLoss

# ────────────────────────────────
# CONFIGURATION
# ────────────────────────────────
CONFIG = {
    "img_dir":             "./test_images",
    "batch_size":          4,
    "num_workers":         0,

    "epochs":              10,
    "initial_radius":      5.0,
    "lr":                  1e-4,
    "weight_decay":        0.0,

    "perceptual_threshold": 0.015,
    "alpha":                2.0,
    "beta_resnet":          0.0,
    "beta_clip":            1.0,
    "penalty_weight":       10.0,

    "sample_interval":     5,
    "sample_size":         2,
    "sample_dir":          "samples",

    # folders for different saves
    "batch_ckpt_dir":      "checkpoints",
    "epoch_dump_dir":      "dumped_models",
    "history_dir":         "model_history",

    # checkpoint frequency (batches)
    "batch_ckpt_interval": 200,

    # optional checkpoint to resume from (raw state_dict or wrapped dict)
    "resume_from":         "./checkpoints/model_epoch003_batch0200.pt",
}


def get_dataloader(cfg):
    ds = SpeciousDataset(cfg["img_dir"], resolution=(224, 224))
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )


def build_model(device, cfg):
    return AdversarialFFTUNet(cutoff_radius=cfg["initial_radius"]).to(device)


def build_loss(device, cfg):
    return SpeciousLoss(
        device=device,
        perceptual_threshold=cfg["perceptual_threshold"],
        alpha=cfg["alpha"],
        beta_resnet=cfg["beta_resnet"],
        beta_clip=cfg["beta_clip"],
        penalty_weight=cfg["penalty_weight"]
    )


def train_epoch(model, loss_fn, optimizer, loader, device, epoch, history, cfg):
    model.train()
    sums = {k: 0.0 for k in ["lpips", "resnet_loss", "clip_loss", "feature_loss", "total_loss"]}
    n = 0

    for batch_idx, imgs in enumerate(loader, start=1):
        imgs = imgs.to(device)
        adv_imgs, _ = model(imgs)

        losses = loss_fn(imgs, adv_imgs)
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        # accumulate for epoch stats
        b = imgs.size(0)
        for k in sums:
            sums[k] += losses[k].item() * b
        n += b

        # record batch metrics
        history.append({
            "epoch":       epoch,
            "batch":       batch_idx,
            **{k: losses[k].item() for k in losses}
        })

        # save checkpoint every cfg["batch_ckpt_interval"] batches
        if batch_idx % cfg["batch_ckpt_interval"] == 0:
            ckpt_dir = Path(cfg["batch_ckpt_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "batch": batch_idx
            }, ckpt_dir / f"model_epoch{epoch:03d}_batch{batch_idx:04d}.pt")

        # print batch stats
        print(
            f"Epoch {epoch}/{cfg['epochs']} | "
            f"Batch {batch_idx}/{len(loader)} | "
            + " | ".join(f"{k} {losses[k]:.4f}" for k in losses)
        )

    epoch_stats = {k: sums[k] / n for k in sums}
    return epoch_stats, history


def save_samples(model, loader, device, cfg, epoch):
    model.eval()
    imgs = next(iter(loader))[: cfg["sample_size"]]
    imgs = imgs.to(device)
    adv_imgs, _ = model(imgs)

    os.makedirs(cfg["sample_dir"], exist_ok=True)
    for i in range(len(imgs)):
        pair = torch.stack([imgs[i].cpu(), adv_imgs[i].cpu()], dim=0)
        save_image(
            pair,
            os.path.join(cfg["sample_dir"], f"epoch{epoch:03d}_sample{i}.png"),
            nrow=2, normalize=True
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cfg = CONFIG

    loader = get_dataloader(cfg)
    model = build_model(device, cfg)
    loss_fn = build_loss(device, cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    # Optionally resume from a prior checkpoint
    resume_path = cfg.get("resume_from", "")
    start_epoch = 1
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        # Handle both raw state_dict and wrapped checkpoint
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
            start_epoch = ckpt.get("epoch", 1) + 1
            print(f"Resumed wrapped checkpoint from epoch {start_epoch-1}")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded raw state_dict from {resume_path}, starting at epoch {start_epoch}")

    history = []

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        stats, history = train_epoch(
            model, loss_fn, optimizer, loader, device, epoch, history, cfg
        )

        # epoch summary
        print(f"*** Epoch {epoch}/{cfg['epochs']} summary ***")
        print(" | ".join(f"{k} {stats[k]:.4f}" for k in stats), "\n")

        # save sample images
        if epoch % cfg["sample_interval"] == 0:
            save_samples(model, loader, device, cfg, epoch)

        # save full-model dump after epoch
        dump_dir = Path(cfg["epoch_dump_dir"])
        dump_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }, dump_dir / f"model_epoch{epoch:03d}.pth")

        # save history CSV (logs every batch)
        hist_dir = Path(cfg["history_dir"])
        hist_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(hist_dir / "training_history.csv", index=False)


if __name__ == "__main__":
    main()

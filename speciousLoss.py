# SpeciousLoss.py

import torch
import torch.nn as nn
import torchvision.models as models
import lpips
import torchvision.transforms as T
import clip
import torch.nn.functional as F

class SpeciousLoss(nn.Module):
    """
    SpeciousLoss combines perceptual LPIPS loss with feature distortion from ResNet50 and CLIP.
    Returns LPIPS loss, ResNet feature loss, CLIP feature loss, combined feature loss, and total loss.
    """
    def __init__(self, device='cpu', perceptual_threshold=0.1,
                alpha=1.0, beta_resnet=0.5, beta_clip=1.0, penalty_weight=10.0):
        super().__init__()
        # Perceptual LPIPS
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        # ResNet50 feature extractor
        resnet = models.resnet50(pretrained=True)
        for p in resnet.parameters():
            p.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
        # CLIP model
        self.clip_model, _ = clip.load('ViT-B/32', device=device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()
        # Hyperparameters
        self.perceptual_threshold = perceptual_threshold
        self.alpha = alpha
        self.beta_resnet = beta_resnet
        self.beta_clip = beta_clip
        self.penalty_weight = penalty_weight

    
    def _extract_resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x_norm = (x - mean) / std
        feats = self.resnet(x_norm)
        return feats.view(feats.size(0), -1)

    
    def _extract_clip_features(self, x: torch.Tensor) -> torch.Tensor:
        resize = T.Resize((224, 224))
        x = resize(x)  # Resize to 224x224
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
        x_norm = (x - mean) / std
        return self.clip_model.encode_image(x_norm)

    def forward(self, orig_rgb: torch.Tensor, adv_rgb: torch.Tensor):
        """
        Args:
            orig_rgb, adv_rgb: tensors (B,3,H,W), values in [0,1]
        Returns:
            dict with keys: lpips, resnet_loss, clip_loss, feature_loss, total_loss
        """
        # LPIPS: inputs scaled to [-1,1]
        orig_norm = orig_rgb * 2 - 1
        adv_norm  = adv_rgb  * 2 - 1
        lpips_val = self.lpips_loss(orig_norm, adv_norm).mean()
        # ResNet features
        res_orig = self._extract_resnet_features(orig_rgb)
        res_adv  = self._extract_resnet_features(adv_rgb)
        resnet_loss = nn.functional.mse_loss(res_orig, res_adv)
        # CLIP features
        clip_orig = self._extract_clip_features(orig_rgb)
        clip_adv  = self._extract_clip_features(adv_rgb)
        clip_loss = nn.functional.mse_loss(clip_orig, clip_adv)
        # Combined feature distortion
        feature_loss = self.beta_resnet * resnet_loss + self.beta_clip * clip_loss
        # Total loss: minimize LPIPS, maximize feature loss
        total = torch.exp(self.alpha * lpips_val - feature_loss)
        # Penalty if perceptual exceeds threshold
        if lpips_val > self.perceptual_threshold:
            penalty = self.penalty_weight * (lpips_val - self.perceptual_threshold)
            total = total + penalty
        return {
            'lpips': lpips_val,
            'resnet_loss': resnet_loss,
            'clip_loss': clip_loss,
            'feature_loss': feature_loss,
            'total_loss': total
        }

# dataset.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class SpeciousDataset(Dataset):
    """
    PyTorch Dataset that loads images from a directory and resizes them to a fixed resolution.
    """
    def __init__(self, img_dir, resolution=(224, 224)):
        """
        Args:
            img_dir (str): Directory containing images.
            resolution (tuple): Target resolution (width, height) for resizing images.
        """
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.resolution = resolution

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image and ensure RGB format
        img = Image.open(self.img_paths[idx]).convert('RGB')

        # Resize to fixed resolution
        img = img.resize(self.resolution, Image.BICUBIC)

        # Convert to NumPy array and normalize to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0  # Shape: (H, W, 3)

        # Convert to PyTorch tensor and permute to (C, H, W)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # Shape: (3, H, W)

        return img_t

# Usage example:
# dataset = SpeciousDataset('./dataset', resolution=(224, 224))
# loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)




import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.degradation import degrade_image


class RestorationDataset(Dataset):
    def __init__(self, clean_dir="sample_images/clean", image_size=256):
        self.image_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in '{clean_dir}'. "
                f"Run generate_dataset.py first, or add your own images there."
            )
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load clean image
        clean_bgr = cv2.imread(self.image_paths[idx])
        clean_bgr = cv2.resize(clean_bgr, (self.image_size, self.image_size))

        # 2. Degrade it, with randomized strength for variety each epoch
        noise_sigma = random.uniform(15, 35)
        blur_kernel = random.choice([3, 5, 7])
        degraded_bgr = degrade_image(clean_bgr, noise_sigma=noise_sigma, blur_kernel=blur_kernel)

        # 3. Convert BGR (OpenCV default) -> RGB, and normalize pixels to [0, 1]
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 4. Convert to PyTorch tensors with shape (Channels, Height, Width)
        clean_tensor = torch.from_numpy(clean_rgb).permute(2, 0, 1)
        degraded_tensor = torch.from_numpy(degraded_rgb).permute(2, 0, 1)

        return degraded_tensor, clean_tensor


if __name__ == "__main__":
    # Quick sanity check
    dataset = RestorationDataset()
    print(f"Dataset size: {len(dataset)} images")

    degraded, clean = dataset[0]
    print("Degraded tensor shape:", degraded.shape)
    print("Clean tensor shape:   ", clean.shape)
    print("Value range (should be ~0 to 1):", degraded.min().item(), "-", degraded.max().item())
    print("PASSED: dataset loads and returns paired tensors.")
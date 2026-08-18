"""

Loads the trained restoration model, runs it on a test image, and reports:
- PSNR (degraded vs clean, restored vs clean)
- SSIM (degraded vs clean, restored vs clean)
- Sharpness improvement (degraded -> restored)

Also saves a side-by-side comparison image: Original | Degraded | Restored
"""

import cv2
import numpy as np
import torch

from model import RestorationAutoencoder
from utils.degradation import degrade_image
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sharpness, sharpness_improvement_percent


MODEL_PATH = "models/restoration_model.pth"
TEST_IMAGE_PATH = "sample_images/clean/chip_000.png"  # change this to test other images
IMAGE_SIZE = 256


def tensor_to_bgr_image(tensor):
    """Convert a model output tensor (C,H,W, range 0-1) back into a normal uint8 BGR image."""
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # -> (H, W, C), RGB, 0-1
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return bgr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- LOAD MODEL ----------
    model = RestorationAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # inference mode: disables training-only behaviors
    print(f"Loaded model from {MODEL_PATH}")

    # ---------- LOAD + DEGRADE TEST IMAGE ----------
    clean_bgr = cv2.imread(TEST_IMAGE_PATH)
    clean_bgr = cv2.resize(clean_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    degraded_bgr = degrade_image(clean_bgr, noise_sigma=25, blur_kernel=7)

    # ---------- RUN THROUGH MODEL ----------
    degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(degraded_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():  # no gradient tracking needed, we're not training
        output_tensor = model(input_tensor)

    restored_bgr = tensor_to_bgr_image(output_tensor)

    # ---------- METRICS ----------
    psnr_degraded = calculate_psnr(clean_bgr, degraded_bgr)
    psnr_restored = calculate_psnr(clean_bgr, restored_bgr)

    ssim_degraded = calculate_ssim(clean_bgr, degraded_bgr)
    ssim_restored = calculate_ssim(clean_bgr, restored_bgr)

    sharp_degraded = calculate_sharpness(degraded_bgr)
    sharp_restored = calculate_sharpness(restored_bgr)
    sharp_improvement = sharpness_improvement_percent(sharp_degraded, sharp_restored)

    # ---------- REPORT ----------
    print("\n--- RESULTS ---")
    print(f"PSNR:  {psnr_degraded:.2f} dB  ->  {psnr_restored:.2f} dB")
    print(f"SSIM:  {ssim_degraded:.3f}      ->  {ssim_restored:.3f}")
    print(f"Sharpness improvement: {sharp_improvement:+.1f}%")

    # ---------- SAVE VISUAL COMPARISON ----------
    comparison = np.hstack([clean_bgr, degraded_bgr, restored_bgr])
    out_path = "outputs/step4_comparison.png"
    cv2.imwrite(out_path, comparison)
    print(f"\nSaved comparison image (Original | Degraded | Restored) to: {out_path}")


if __name__ == "__main__":
    main()
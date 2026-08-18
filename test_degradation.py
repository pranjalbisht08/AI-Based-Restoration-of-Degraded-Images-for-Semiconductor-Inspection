import cv2
import numpy as np
import os
from utils.degradation import degrade_image


def make_synthetic_chip_image(size=256):
    """
    Creates a fake semiconductor/circuit-like pattern:
    grid lines + circular pads on a gray background.
    This stands in for a real inspection image until you have one.
    """
    img = np.full((size, size, 3), 200, dtype=np.uint8)  # light gray background

    # Draw grid lines (like circuit traces)
    for i in range(0, size, 20):
        cv2.line(img, (i, 0), (i, size), (60, 60, 60), 1)
        cv2.line(img, (0, i), (size, i), (60, 60, 60), 1)

    # Draw circular "pads" (like bond pads on a chip)
    for cx in range(30, size, 50):
        for cy in range(30, size, 50):
            cv2.circle(img, (cx, cy), 8, (20, 20, 20), -1)

    # Draw a few thicker "wafer edge" style rectangles
    cv2.rectangle(img, (10, 10), (size - 10, size - 10), (0, 0, 0), 2)

    return img


def main():
    os.makedirs("sample_images/clean", exist_ok=True)
    os.makedirs("sample_images/degraded", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Create a clean synthetic image
    clean = make_synthetic_chip_image()
    cv2.imwrite("sample_images/clean/sample_clean.png", clean)

    # 2. Degrade it
    degraded = degrade_image(clean, noise_sigma=25, blur_kernel= 7)
    cv2.imwrite("sample_images/degraded/sample_degraded.png", degraded)

    # 3. Save a side-by-side comparison for easy viewing
    comparison = np.hstack([clean, degraded])
    cv2.imwrite("outputs/step1_comparison.png", comparison)

    print("Done!")
    print("Clean image saved to:      sample_images/clean/sample_clean.png")
    print("Degraded image saved to:   sample_images/degraded/sample_degraded.png")
    print("Side-by-side comparison:   outputs/step1_comparison.png")


if __name__ == "__main__":
    main()
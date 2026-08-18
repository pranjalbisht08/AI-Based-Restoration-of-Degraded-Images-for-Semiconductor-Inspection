import cv2
import numpy as np
import os
import random


def make_synthetic_chip_image(size=256, grid_spacing=20, pad_radius=8, seed=None):
    """
    Creates one fake circuit/wafer-like pattern image.
    Parameters are randomized slightly per image to create dataset variety.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    bg_value = random.randint(180, 220)
    img = np.full((size, size, 3), bg_value, dtype=np.uint8)

    # Grid lines (circuit traces)
    for i in range(0, size, grid_spacing):
        cv2.line(img, (i, 0), (i, size), (60, 60, 60), 1)
        cv2.line(img, (0, i), (size, i), (60, 60, 60), 1)

    # Circular pads, spaced out, with slight random jitter
    step = grid_spacing * 2 + random.randint(-5, 10)
    step = max(step, 20)
    offset = random.randint(20, 40)
    for cx in range(offset, size, step):
        for cy in range(offset, size, step):
            jitter_x = random.randint(-3, 3)
            jitter_y = random.randint(-3, 3)
            cv2.circle(img, (cx + jitter_x, cy + jitter_y), pad_radius, (20, 20, 20), -1)

    # Border (wafer edge)
    cv2.rectangle(img, (10, 10), (size - 10, size - 10), (0, 0, 0), 2)

    return img


def main(num_images=80, out_dir="sample_images/clean"):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_images):
        grid_spacing = random.randint(15, 28)
        pad_radius = random.randint(5, 10)
        img = make_synthetic_chip_image(
            size=256,
            grid_spacing=grid_spacing,
            pad_radius=pad_radius,
            seed=i,
        )
        filename = os.path.join(out_dir, f"chip_{i:03d}.png")
        cv2.imwrite(filename, img)

    print(f"Generated {num_images} synthetic clean images in '{out_dir}/'")


if __name__ == "__main__":
    main()
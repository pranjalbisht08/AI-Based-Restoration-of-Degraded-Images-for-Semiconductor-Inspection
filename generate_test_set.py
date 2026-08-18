"""
generate_test_set.py
---------------------
Creates a small set of DEGRADED test images for submission purposes.

Takes clean images from sample_images/clean/, degrades them (same way as
training), and saves them to test_images/. This is the "test set" you run
inference.py on to produce your Restored Test Outputs folder.
"""

import cv2
import os
import glob
import random

from utils.degradation import degrade_image

CLEAN_DIR = "sample_images/clean"
TEST_DIR = "test_images"
NUM_TEST_IMAGES = 15
IMAGE_SIZE = 256


def main():
    os.makedirs(TEST_DIR, exist_ok=True)

    clean_paths = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.png")))
    if len(clean_paths) == 0:
        print(f"ERROR: No clean images found in '{CLEAN_DIR}'. Run generate_dataset.py first.")
        return

    random.seed(123)  # fixed seed so the test set is reproducible
    chosen_paths = random.sample(clean_paths, min(NUM_TEST_IMAGES, len(clean_paths)))

    for path in chosen_paths:
        clean = cv2.imread(path)
        clean = cv2.resize(clean, (IMAGE_SIZE, IMAGE_SIZE))

        degraded = degrade_image(clean, noise_sigma=25, blur_kernel=7)

        filename = os.path.basename(path)
        out_path = os.path.join(TEST_DIR, filename)
        cv2.imwrite(out_path, degraded)

    print(f"Generated {len(chosen_paths)} degraded test images in '{TEST_DIR}/'")


if __name__ == "__main__":
    main()
    
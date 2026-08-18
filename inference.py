"""
inference.py
-------------
Standalone evaluation/inference script for submission.

Loads the trained restoration model and runs it on every image in an input
directory, saving the restored version of each to an output directory.

USAGE:
    python inference.py --input_dir path/to/test_images --output_dir path/to/results

Requirements this satisfies:
- Accepts input directory and output directory as command-line arguments.
- Loads the trained model.
- Runs inference on all images in the input directory.
- Writes restored outputs to the output directory.
- Runs with no manual code edits.
"""

import argparse
import os
import glob
import sys

import cv2
import numpy as np
import torch

from model import RestorationAutoencoder

MODEL_PATH = "models/restoration_model.pth"
IMAGE_SIZE = 256
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".npy")


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Train the model first by running: python train.py")
        sys.exit(1)

    model = RestorationAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_image_any(path):
    """
    Loads an image from disk as a BGR uint8 NumPy array (our internal format),
    regardless of whether the file is a standard image (.png/.jpg/.bmp) or a
    raw NumPy array (.npy).

    .npy files are assumed to store pixel data in RGB channel order (the
    common NumPy/PIL convention), as either:
    - uint8 values in range 0-255, or
    - float values in range 0-1 (normalized)
    - grayscale (H, W) or color (H, W, 3) / (H, W, 4)
    All are normalized here to a standard BGR uint8 image for processing.
    """
    if path.lower().endswith(".npy"):
        array = np.load(path)

        # Normalize dtype/range to uint8 0-255
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255.0).clip(0, 255).astype(np.uint8)
            else:
                array = array.clip(0, 255).astype(np.uint8)

        # Normalize channels to 3-channel BGR
        if array.ndim == 2:
            bgr = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 4:
            bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        elif array.ndim == 3 and array.shape[2] == 3:
            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported .npy array shape {array.shape} in '{path}'")

        return bgr
    else:
        return cv2.imread(path)


def save_image_any(path, bgr_image):
    """
    Saves a BGR uint8 image to disk. If the target path ends in .npy, saves
    as a raw NumPy array in RGB channel order, uint8, shape (H, W, 3) --
    matching the convention assumed in load_image_any. Otherwise saves as a
    standard image file (png/jpg/etc, inferred from the extension).
    """
    if path.lower().endswith(".npy"):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        np.save(path, rgb_image)
    else:
        cv2.imwrite(path, bgr_image)


def restore_image(model, device, image_bgr):
    """Run a single BGR image through the model and return the restored BGR image,
    resized back to the original input dimensions."""
    original_h, original_w = image_bgr.shape[:2]

    resized = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Resize back to the original image's dimensions
    restored_bgr = cv2.resize(restored_bgr, (original_w, original_h))
    return restored_bgr


def main():
    parser = argparse.ArgumentParser(
        description="Run trained restoration model on a directory of test images."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to directory containing input (degraded/test) images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to directory where restored images will be saved."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found: '{args.input_dir}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    print(f"Loaded model from '{MODEL_PATH}'")

    # Collect all valid image files in the input directory
    image_paths = []
    for ext in VALID_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext.upper()}")))
    image_paths = sorted(set(image_paths))

    if len(image_paths) == 0:
        print(f"WARNING: No images found in '{args.input_dir}'.")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s). Running inference...")

    processed_count = 0
    for path in image_paths:
        try:
            image_bgr = load_image_any(path)
        except Exception as e:
            print(f"  Skipping unreadable file: {path} ({e})")
            continue

        if image_bgr is None:
            print(f"  Skipping unreadable file: {path}")
            continue

        restored_bgr = restore_image(model, device, image_bgr)

        filename = os.path.basename(path)
        out_path = os.path.join(args.output_dir, filename)  # same filename+extension as input
        save_image_any(out_path, restored_bgr)

        processed_count += 1
        print(f"  [{processed_count}/{len(image_paths)}] {filename} -> {out_path}")

    print(f"\nDone. {processed_count} restored image(s) saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()

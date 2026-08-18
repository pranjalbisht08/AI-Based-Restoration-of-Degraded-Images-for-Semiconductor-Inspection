
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
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Train the model first by running: python train.py")
        sys.exit(1)

    model = RestorationAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


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
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"  Skipping unreadable file: {path}")
            continue

        restored_bgr = restore_image(model, device, image_bgr)

        filename = os.path.basename(path)
        out_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(out_path, restored_bgr)

        processed_count += 1
        print(f"  [{processed_count}/{len(image_paths)}] {filename} -> {out_path}")

    print(f"\nDone. {processed_count} restored image(s) saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
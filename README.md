
# SEMICON AI — AI-Based Restoration of Degraded Images for Semiconductor Inspection

Semicon India Hackathon 2026

An AI-powered system that restores degraded (noisy/blurred) semiconductor
inspection images, so that structural/pattern information is preserved for
downstream defect inspection. Includes a trained convolutional autoencoder,
quantitative evaluation (PSNR/SSIM/sharpness), an interactive Streamlit
dashboard, and a classical-CV defect detection module.

---

## 1. Repository Structure

```
semicon-restoration/
├── app.py                     # Streamlit interactive dashboard
├── model.py                   # Autoencoder model definition
├── train.py                   # Training script (reproduces the model from scratch)
├── inference.py                # Standalone evaluation script (dir -> dir)
├── evaluate.py                 # Single-image evaluation with PSNR/SSIM/sharpness report
├── generate_dataset.py         # Generates the synthetic training dataset
├── generate_test_set.py        # Generates a degraded test set for inference.py
├── requirements.txt
├── models/
│   └── restoration_model.pth   # Trained model weights
├── utils/
│   ├── degradation.py          # Noise/blur simulation
│   ├── metrics.py               # PSNR, SSIM, sharpness
│   └── defect_detection.py      # Golden-template defect detection
├── sample_images/
│   └── clean/                   # Synthetic training images
├── test_images/                 # Degraded test set (generated)
└── restored_outputs/            # Model outputs on the test set (generated)
```

---

## 2. Requirements

- Python 3.9–3.12 recommended
- pip

Install all dependencies:

```bash
pip install -r requirements.txt
```

> Note: This project was developed and run directly with a global Python
> installation (no virtual environment). A virtual environment is optional
> but recommended for isolation:
> ```bash
> python -m venv venv
> venv\Scripts\activate        # Windows
> source venv/bin/activate     # Mac/Linux
> pip install -r requirements.txt
> ```

---

## 3. Reproducing the Model From Scratch

**Step 1 — Generate the training dataset** (synthetic chip-pattern images):
```bash
python generate_dataset.py
```
This creates `sample_images/clean/` with 80 synthetic clean images.

**Step 2 — Train the model:**
```bash
python train.py
```
This trains a convolutional autoencoder for 80 epochs and saves the trained
weights to `models/restoration_model.pth`. Training runs on CPU in a few
minutes (GPU used automatically if available).

---

## 4. Running Inference (Evaluation Script)

The standalone evaluation script accepts an input directory of test images
and an output directory, and writes restored versions of each image to the
output directory. No manual code edits required.

**Optional — generate a sample degraded test set first:**
```bash
python generate_test_set.py
```
This creates `test_images/` with 15 degraded sample images.

**Run inference:**
```bash
python inference.py --input_dir test_images --output_dir restored_outputs
```

This will:
1. Load the trained model from `models/restoration_model.pth`
2. Run every image in `--input_dir` through the model
3. Save the restored version of each image (at its original resolution) to `--output_dir`

To run on your own test images instead, just point `--input_dir` at any
folder of images:
```bash
python inference.py --input_dir path/to/your/images --output_dir path/to/save/results
```

---

## 5. Quantitative Evaluation (Metrics)

To see PSNR / SSIM / sharpness metrics on a single test image, with a
side-by-side visual comparison:
```bash
python evaluate.py
```
This uses `sample_images/clean/chip_000.png` as the test image by default
(edit `TEST_IMAGE_PATH` at the top of `evaluate.py` to change it), and saves
a comparison image to `outputs/step4_comparison.png`.

**Typical results on the synthetic test set:**

| Metric | Degraded | Restored |
|---|---|---|
| PSNR | ~14 dB | ~20 dB |
| SSIM | ~0.35 | ~0.84 |
| Sharpness | baseline | +68% |

---

## 6. Interactive Dashboard

Launch the Streamlit demo dashboard:
```bash
streamlit run app.py
```
This opens a browser dashboard where you can:
- Select a bundled sample image or upload your own
- Optionally inject a simulated defect
- Simulate degradation (noise + blur)
- Run AI restoration
- View PSNR/SSIM/sharpness metrics
- Run golden-template defect detection

---

## 7. Known Limitations

- **Domain-specific model:** The model is trained only on synthetic
  chip-pattern images (grid lines + circular pads). It restores images in
  this style well, but does not generalize to arbitrary photos outside this
  distribution (a known generalization gap). The dashboard displays a
  warning when a user uploads an image outside this style.
- **Defect detection is classical CV, not deep learning:** Golden-template
  comparison (pixel-difference thresholding) is used instead of a trained
  defect classifier, since no labeled real-world defect dataset was
  available on this timeline. This is a standard, explainable technique
  already used in real PCB/wafer inspection.
- **Synthetic dataset:** Training data is procedurally generated rather than
  real semiconductor inspection imagery, due to data availability
  constraints. The pipeline (degradation simulation, model, training,
  evaluation) is designed to drop in real paired data with no code changes.

---

## 8. Tech Stack

Python, PyTorch, OpenCV, scikit-image, NumPy, Pillow, Streamlit

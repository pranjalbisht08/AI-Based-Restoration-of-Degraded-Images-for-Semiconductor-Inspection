import streamlit as st
import numpy as np
import cv2
import torch
import glob
import os
from PIL import Image

from model import RestorationAutoencoder
from utils.degradation import degrade_image
from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_sharpness,
    sharpness_improvement_percent,
)
from utils.defect_detection import simulate_defect, detect_defects

MODEL_PATH = "models/restoration_model.pth"
SAMPLE_DIR = "sample_images/clean"
IMAGE_SIZE = 256


# ---------- MODEL LOADING (cached so it only loads once, not on every rerun) ----------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestorationAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


# ---------- HELPER FUNCTIONS ----------
def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert an uploaded PIL image to a resized BGR NumPy array (our internal format)."""
    rgb = np.array(pil_image.convert("RGB"))
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_display(bgr_image: np.ndarray) -> np.ndarray:
    """Convert our internal BGR format back to RGB for correct display in Streamlit."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def run_restoration(degraded_bgr: np.ndarray) -> np.ndarray:
    """Run the degraded image through the trained model and return a restored BGR image."""
    model, device = load_model()

    rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return restored_bgr


def get_sample_images():
    """Return a sorted list of sample image paths bundled with the project."""
    return sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SEMICON AI", page_icon="🔬", layout="wide")

# ---------- CUSTOM STYLING ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #101c2c 0%, #060a12 55%, #04060a 100%);
    }

    /* Header banner */
    .semicon-header {
        padding: 2.6rem 2.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(0,90,255,0.05));
        border: 1px solid rgba(0,212,255,0.25);
        margin-bottom: 1.5rem;
    }
    .semicon-title {
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        line-height: 1.1;
        background: linear-gradient(90deg, #00e5ff, #4f8cff 60%, #7cffcb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .semicon-subtitle {
        color: #9fb3c8;
        font-size: 1.35rem;
        margin-top: 0.6rem;
    }
    .semicon-tag {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #00e5ff;
        background: rgba(0,229,255,0.08);
        border: 1px solid rgba(0,229,255,0.3);
        padding: 4px 12px;
        border-radius: 20px;
        margin-top: 1rem;
        letter-spacing: 0.05em;
    }

    /* Section headers */
    h3, .stSubheader {
        color: #e6f1ff !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0072ff, #00c6ff);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 2px 12px rgba(0,150,255,0.25);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 18px rgba(0,180,255,0.4);
        color: white;
    }
    .stButton>button:disabled {
        background: #1c2635;
        color: #55647a;
        box-shadow: none;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0,229,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(0,229,255,0.18);
        border-radius: 12px;
        padding: 1rem 1rem 0.6rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #9fb3c8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #00e5ff !important;
    }

    /* Image panel captions */
    .panel-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        color: #7cffcb;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="semicon-header">
        <p class="semicon-title">🔬 SEMICON AI</p>
        <p class="semicon-subtitle">AI-Based Image Restoration for Semiconductor Inspection</p>
        <span class="semicon-tag">SEMICON INDIA HACKATHON 2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload a semiconductor/wafer inspection image, simulate real-world degradation "
    "(noise + blur), then restore it using a trained AI model."
)

# ---------- SAFETY CHECK: model file must exist ----------
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at '{MODEL_PATH}'. "
        f"Run `python train.py` first to train and save the model, then restart this app."
    )
    st.stop()

# ---------- SESSION STATE INIT ----------
if "golden_image" not in st.session_state:
    st.session_state.golden_image = None  # defect-free reference, for comparison later
if "clean_image" not in st.session_state:
    st.session_state.clean_image = None  # what actually gets degraded (golden + optional defect)
if "degraded_image" not in st.session_state:
    st.session_state.degraded_image = None
if "restored_image" not in st.session_state:
    st.session_state.restored_image = None
if "defect_type" not in st.session_state:
    st.session_state.defect_type = None

# ---------- IMAGE SOURCE: upload or pick a bundled sample ----------
source_mode = st.radio(
    "Choose an image source",
    ["Use a sample image", "Upload my own"],
    horizontal=True,
)

new_clean = None

if source_mode == "Upload my own":
    uploaded_file = st.file_uploader("Upload an inspection image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            new_clean = pil_to_bgr(pil_image)
        except Exception as e:
            st.error(f"Could not read that image file ({e}). Please try a different file.")

else:
    sample_paths = get_sample_images()
    if len(sample_paths) == 0:
        st.warning(
            f"No sample images found in '{SAMPLE_DIR}/'. "
            f"Run `python generate_dataset.py` first, or switch to 'Upload my own'."
        )
    else:
        sample_names = [os.path.basename(p) for p in sample_paths]
        chosen_name = st.selectbox("Pick a sample image", sample_names)
        chosen_path = os.path.join(SAMPLE_DIR, chosen_name)
        sample_bgr = cv2.imread(chosen_path)
        if sample_bgr is not None:
            new_clean = cv2.resize(sample_bgr, (IMAGE_SIZE, IMAGE_SIZE))

if new_clean is not None:
    # If the image actually changed, reset downstream results
    if st.session_state.golden_image is None or not np.array_equal(new_clean, st.session_state.golden_image):
        st.session_state.golden_image = new_clean
        st.session_state.clean_image = new_clean
        st.session_state.degraded_image = None
        st.session_state.restored_image = None
        st.session_state.defect_type = None

# ---------- OPTIONAL: INJECT A SIMULATED DEFECT ----------
if st.session_state.golden_image is not None:
    inject_defect = st.checkbox("Inject a simulated defect (for demo)")

    if inject_defect and st.session_state.defect_type is None:
        defected, defect_type = simulate_defect(st.session_state.golden_image)
        st.session_state.clean_image = defected
        st.session_state.defect_type = defect_type
        st.session_state.degraded_image = None
        st.session_state.restored_image = None
        st.rerun()

    if not inject_defect and st.session_state.defect_type is not None:
        st.session_state.clean_image = st.session_state.golden_image
        st.session_state.defect_type = None
        st.session_state.degraded_image = None
        st.session_state.restored_image = None
        st.rerun()

    if st.session_state.defect_type is not None:
        st.caption(f"Simulated defect type: **{st.session_state.defect_type}**")

# ---------- ACTION BUTTONS ----------
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    degrade_clicked = st.button(
        "🌫️ Simulate Degradation", disabled=(st.session_state.clean_image is None)
    )

with col_btn2:
    restore_clicked = st.button(
        "✨ AI Restore", disabled=(st.session_state.degraded_image is None)
    )

with col_btn3:
    reset_clicked = st.button("🔄 Reset")

if reset_clicked:
    st.session_state.golden_image = None
    st.session_state.clean_image = None
    st.session_state.degraded_image = None
    st.session_state.restored_image = None
    st.session_state.defect_type = None
    st.rerun()

if degrade_clicked and st.session_state.clean_image is not None:
    st.session_state.degraded_image = degrade_image(
        st.session_state.clean_image, noise_sigma=25, blur_kernel=7
    )
    st.session_state.restored_image = None  # reset restoration if we re-degrade
    st.rerun()

if restore_clicked and st.session_state.degraded_image is not None:
    st.session_state.restored_image = run_restoration(st.session_state.degraded_image)
    st.rerun()

# ---------- IMAGE DISPLAY ----------
st.markdown("---")
st.subheader("🖼️ Visual Comparison")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown('<p class="panel-label">📥 Original</p>', unsafe_allow_html=True)
        if st.session_state.clean_image is not None:
            st.image(bgr_to_display(st.session_state.clean_image), use_container_width=True)
        else:
            st.info("Upload an image to begin.")

with col2:
    with st.container(border=True):
        st.markdown('<p class="panel-label">🌫️ Degraded</p>', unsafe_allow_html=True)
        if st.session_state.degraded_image is not None:
            st.image(bgr_to_display(st.session_state.degraded_image), use_container_width=True)
        else:
            st.info("Click 'Simulate Degradation'.")

with col3:
    with st.container(border=True):
        st.markdown('<p class="panel-label">✨ AI Restored</p>', unsafe_allow_html=True)
        if st.session_state.restored_image is not None:
            st.image(bgr_to_display(st.session_state.restored_image), use_container_width=True)
        else:
            st.info("Click 'AI Restore'.")

# ---------- METRICS ----------
if st.session_state.restored_image is not None:
    st.markdown("---")
    st.subheader("📈 Performance Metrics")

    clean = st.session_state.clean_image
    degraded = st.session_state.degraded_image
    restored = st.session_state.restored_image

    psnr_degraded = calculate_psnr(clean, degraded)
    psnr_restored = calculate_psnr(clean, restored)

    ssim_degraded = calculate_ssim(clean, degraded)
    ssim_restored = calculate_ssim(clean, restored)

    sharp_degraded = calculate_sharpness(degraded)
    sharp_restored = calculate_sharpness(restored)
    sharp_improvement = sharpness_improvement_percent(sharp_degraded, sharp_restored)

    m1, m2, m3 = st.columns(3)
    m1.metric("📊 PSNR", f"{psnr_restored:.2f} dB", f"+{psnr_restored - psnr_degraded:.2f} dB")
    m2.metric("🧩 SSIM", f"{ssim_restored:.3f}", f"+{ssim_restored - ssim_degraded:.3f}")
    m3.metric("🔎 Sharpness Improvement", f"{sharp_improvement:+.1f}%")

    st.markdown("---")
    st.subheader("🧪 Inspection Result")
    st.write(
        "Restoration aims to preserve the structural and pattern information "
        "(traces, pads, edges) needed for downstream semiconductor defect inspection, "
        "even when the input image is affected by sensor noise or focus blur."
    )

    # ---------- OPTIONAL: DEFECT DETECTION ----------
    st.markdown("---")
    st.subheader("🎯 Defect Detection (Optional)")
    st.write(
        "Compares the restored image against the known-good golden reference "
        "to automatically locate anomalies -- a standard technique in PCB/wafer inspection."
    )

    detect_clicked = st.button("🎯 Detect Defects")

    if detect_clicked:
        annotated, boxes = detect_defects(st.session_state.golden_image, restored)

        d1, d2 = st.columns(2)
        with d1:
            with st.container(border=True):
                st.markdown('<p class="panel-label">🏅 Golden Reference</p>', unsafe_allow_html=True)
                st.image(bgr_to_display(st.session_state.golden_image), use_container_width=True)
        with d2:
            with st.container(border=True):
                st.markdown('<p class="panel-label">🚨 Detected Defects</p>', unsafe_allow_html=True)
                st.image(bgr_to_display(annotated), use_container_width=True)

        if len(boxes) > 0:
            st.error(f"⚠️ {len(boxes)} defect region(s) detected.")
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                st.write(f"Defect {i}: location ({x}, {y}), size {w}x{h} px")
        else:
            st.success("✅ No significant defects detected.")
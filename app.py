import os
import sys
import time
import io
from pathlib import Path

# Ensure local dependencies are recognized
site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.model import DeblurNetLite, count_parameters

# Page Configuration
st.set_page_config(
    page_title="AI Image Deblurring | DeblurNet-Lite",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #2563eb;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_deblur_model(checkpoint_path="checkpoints/best_model.pth"):
    """Loads and caches the trained DeblurNet-Lite model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = Path(checkpoint_path)
    
    if not ckpt_file.exists():
        return None, device, "Model checkpoint not found. Please train the model first!"
        
    try:
        model = DeblurNetLite().to(device)
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, device, None
    except Exception as e:
        return None, device, f"Error loading checkpoint: {str(e)}"


def restore_image(model, device, image: Image.Image, sharpness_boost: float = 1.0):
    """
    Restores blurry images using trained DeblurNet-Lite with Global Residual Learning.
    Matches receptive field scale for optimal edge recovery.
    """
    orig_w, orig_h = image.size
    
    # Process at model native receptive scale (128, 128)
    proc_img = TF.resize(image, (128, 128), interpolation=TF.InterpolationMode.BICUBIC)
    input_tensor = TF.to_tensor(proc_img).unsqueeze(0).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        enc1 = model.encoder_conv1(input_tensor)
        enc2 = model.encoder_down(enc1)
        bottleneck = model.bottleneck(enc2)
        dec = model.decoder_up(bottleneck)
        fused = torch.cat([dec, enc1], dim=1)
        fused = model.fusion(fused)
        delta = model.output_conv(fused)
        
        # Apply Global Residual sharpening
        restored = torch.clamp(input_tensor + sharpness_boost * delta, 0.0, 1.0)
        
    inference_time = (time.time() - t0) * 1000.0  # in milliseconds
    
    # Convert restored tensor to PIL image
    output_tensor = restored.squeeze(0).cpu().clamp(0.0, 1.0)
    restored_img = TF.to_pil_image(output_tensor)
    
    # Resize back to original user dimensions if uploaded at a different size
    if (orig_w, orig_h) != (128, 128):
        restored_final = restored_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    else:
        restored_final = restored_img
        
    return restored_final, inference_time


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Project Controls")
    
    sharpness = st.slider("Deblurring Boost Factor:", min_value=0.8, max_value=2.0, value=1.0, step=0.1)
    
    st.divider()
    st.markdown("""
    **Project Title:**  
    AI-Based Image Deblurring Using a Custom Deep Learning Model
    
    **Architecture:**  
    DeblurNet-Lite (Global Residual CNN)
    
    **Model Training:**  
    100% Trained from Scratch (No Pretrained Weights)
    
    **Hardware Mode:**  
    CPU-Optimized Inference
    
    **Team Members:**  
    - Student 1: Architecture & Model Design  
    - Student 2: Data Preprocessing & Training  
    - Student 3: Evaluation, Frontend & Defense  
    """)
    st.divider()
    st.info("💡 **Academic Note:** This model uses Global Residual Learning (Output = Input + Δ) to preserve base structure while restoring high-frequency sharp edges.")


# ---------------- MAIN UI ----------------
st.markdown('<div class="main-header">✨ AI Image Deblurring Application</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Restore motion-blurred and out-of-focus images using our custom-trained <b>DeblurNet-Lite</b> neural network.</div>', unsafe_allow_html=True)

# Load Model
model, device, error_msg = load_deblur_model()

if error_msg:
    st.error(f"⚠️ {error_msg}")
    st.stop()

# Input Selection: Upload or Use Test Sample
col_opt1, col_opt2 = st.columns([2, 1])
with col_opt1:
    uploaded_file = st.file_uploader("Upload a blurry image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])
with col_opt2:
    st.write("Or choose a preloaded test sample:")
    test_samples_dir = Path("data/test/blurry")
    sample_options = ["None"]
    if test_samples_dir.exists():
        sample_options += sorted([f.name for f in test_samples_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])[:6]
    selected_sample = st.selectbox("Select demo sample:", sample_options)

input_image = None
image_source_name = ""

if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_source_name = uploaded_file.name
    except Exception as e:
        st.error(f"Failed to load uploaded image: {e}")
elif selected_sample != "None":
    sample_path = test_samples_dir / selected_sample
    input_image = Image.open(sample_path).convert("RGB")
    image_source_name = selected_sample

# Processing and Display
if input_image is not None:
    st.divider()
    
    with st.spinner("Processing image through DeblurNet-Lite..."):
        restored_img, inf_ms = restore_image(model, device, input_image, sharpness_boost=sharpness)
        
    # Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Processing Time", f"{inf_ms:.1f} ms")
    with m2:
        st.metric("Model Parameters", f"{count_parameters(model):,}")
    with m3:
        st.metric("Resolution", f"{input_image.size[0]} x {input_image.size[1]}")
    with m4:
        st.metric("Execution Device", str(device).upper())

    # Visual Comparison Side by Side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Input Blurry Image")
        st.image(input_image, use_container_width=True, caption=f"Source: {image_source_name}")
        
    with col2:
        st.subheader("🌟 DeblurNet-Lite Restored Image")
        st.image(restored_img, use_container_width=True, caption="Restored by Custom Model")

    # Download Option
    st.divider()
    buf = io.BytesIO()
    restored_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    d_col1, d_col2, d_col3 = st.columns([1, 2, 1])
    with d_col2:
        st.download_button(
            label="⬇️ Download Restored Image",
            data=byte_im,
            file_name=f"restored_{Path(image_source_name).stem}.png",
            mime="image/png"
        )
else:
    st.info("👆 Please upload an image or select a preloaded sample above to demonstrate the deblurring model.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>College Group Project: AI-Based Image Deblurring Using a Custom Deep Learning Model | PyTorch & Streamlit</div>", unsafe_allow_html=True)

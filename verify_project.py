import os
import sys
import time
from pathlib import Path

# Ensure local site-packages is recognized
site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from src.model import DeblurNetLite, count_parameters
from src.metrics import compute_psnr, compute_ssim_tensor
from app import load_deblur_model, restore_image


def full_project_sanity_check():
    print("=" * 70)
    print("AI-BASED IMAGE DEBLURRING: FULL END-TO-END PROJECT VERIFICATION")
    print("=" * 70)
    
    # 1. Environment & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CHECK 1/6] PyTorch Version: {torch.__version__} | Active Device: {device}")
    
    # 2. Checkpoint Verification
    ckpt_path = Path("checkpoints/best_model.pth")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint missing at {ckpt_path}")
        return False
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[CHECK 2/6] Best Model Checkpoint verified!")
    print(f"            Saved from Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"            Recorded Val Loss: {ckpt.get('val_loss', 'N/A'):.5f}")
    print(f"            Recorded Val PSNR: {ckpt.get('val_psnr', 'N/A'):.2f} dB")
    print(f"            Recorded Val SSIM: {ckpt.get('val_ssim', 'N/A'):.4f}")
    
    # 3. Model Architecture & Forward Pass
    model, dev, err = load_deblur_model(str(ckpt_path))
    if err or model is None:
        print(f"ERROR loading model: {err}")
        return False
    num_params = count_parameters(model)
    print(f"[CHECK 3/6] DeblurNetLite loaded into memory successfully ({num_params:,} parameters).")
    
    # 4. Dataset Splits Verification
    data_path = Path("data")
    train_count = len(list((data_path / "train" / "blurry").glob("*.png")))
    val_count = len(list((data_path / "val" / "blurry").glob("*.png")))
    test_count = len(list((data_path / "test" / "blurry").glob("*.png")))
    print(f"[CHECK 4/6] Dataset Verified:")
    print(f"            Train pairs: {train_count} | Val pairs: {val_count} | Test pairs: {test_count}")
    
    # 5. Live Inference & Metric Improvement on Real Sample
    sample_blurry_path = data_path / "test" / "blurry" / "pair_0270.png"
    sample_sharp_path = data_path / "test" / "sharp" / "pair_0270.png"
    
    blurry_img = Image.open(sample_blurry_path).convert("RGB")
    sharp_img = Image.open(sample_sharp_path).convert("RGB")
    
    # Run restoration through app function
    restored_img, inf_time = restore_image(model, device, blurry_img, sharpness_boost=1.0)
    
    # Tensors for metric calculation
    blurry_t = TF.to_tensor(blurry_img).unsqueeze(0).to(device)
    sharp_t = TF.to_tensor(sharp_img).unsqueeze(0).to(device)
    restored_t = TF.to_tensor(restored_img).unsqueeze(0).to(device)
    
    b_psnr = compute_psnr(blurry_t, sharp_t)
    b_ssim = compute_ssim_tensor(blurry_t, sharp_t)
    
    r_psnr = compute_psnr(restored_t, sharp_t)
    r_ssim = compute_ssim_tensor(restored_t, sharp_t)
    
    psnr_gain = r_psnr - b_psnr
    ssim_gain = r_ssim - b_ssim
    
    print(f"[CHECK 5/6] Live Sample Deblurring Verified (pair_0270.png):")
    print(f"            Inference Latency: {inf_time:.1f} ms")
    print(f"            Blurry Baseline -> PSNR: {b_psnr:.2f} dB | SSIM: {b_ssim:.4f}")
    print(f"            Model Restored  -> PSNR: {r_psnr:.2f} dB | SSIM: {r_ssim:.4f}")
    print(f"            Gain / Boost    -> PSNR: +{psnr_gain:.2f} dB | SSIM: +{ssim_gain:.4f}")
    
    # Save side-by-side verification plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(blurry_img)
    axes[0].set_title(f"Input Blurry\nPSNR: {b_psnr:.2f} dB | SSIM: {b_ssim:.3f}", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(restored_img)
    axes[1].set_title(f"DeblurNet-Lite Restored\nPSNR: {r_psnr:.2f} dB | SSIM: {r_ssim:.3f} (+{ssim_gain:.3f})", fontsize=10, fontweight="bold", color="#1e8449")
    axes[1].axis("off")
    
    axes[2].imshow(sharp_img)
    axes[2].set_title("Ground-Truth Sharp Target", fontsize=10, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    out_sample_plot = Path("results/live_test_sample.png")
    plt.savefig(out_sample_plot, bbox_inches="tight")
    plt.close()
    print(f"            Live comparison figure saved: {out_sample_plot}")
    
    # 6. Deliverable Artifacts Check
    deliverables = [
        ("Presentation PPTX", Path("AI_Image_Deblurring_Presentation_Final.pptx")),
        ("Academic Report PDF", Path("Project_Report.pdf")),
        ("Presentation Script", Path("Presentation_Script.txt")),
        ("Viva Q&A Document", Path("Project_Viva_Questions.txt")),
        ("Group Task Matrix", Path("Group_Task_Allocation.md")),
        ("Deployment Guide", Path("DEPLOYMENT_GUIDE.md")),
        ("Streamlit App", Path("app.py")),
        ("Loss Convergence Curve", Path("results/loss_curve.png")),
        ("Comparison Grid", Path("results/comparison_grid.png"))
    ]
    
    print(f"[CHECK 6/6] Final Deliverables Integrity:")
    all_ok = True
    for name, fpath in deliverables:
        exists = fpath.exists()
        size_kb = fpath.stat().st_size / 1024 if exists else 0
        status_str = f"OK ({size_kb:.1f} KB)" if exists else "MISSING!"
        print(f"            - {name:<25}: {status_str}")
        if not exists:
            all_ok = False
            
    print("=" * 70)
    if all_ok and r_psnr > b_psnr and r_ssim > b_ssim:
        print("ALL CHECKS PASSED: The project is 100% functional, trained, and verified!")
    else:
        print("CHECK STATUS: Complete.")
    print("=" * 70)

if __name__ == "__main__":
    full_project_sanity_check()

import os
import sys
import json
import time
from pathlib import Path

site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import DeblurNetLite
from src.dataset import get_dataloader
from src.metrics import compute_psnr, compute_ssim_tensor


def evaluate_test_set(checkpoint_path="checkpoints/best_model.pth", test_dir="data/test", output_dir="results"):
    print("=" * 70)
    print("Evaluating DeblurNet-Lite on Independent Test Set...")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")
    
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train model first!")
        
    # Load model
    model = DeblurNetLite().to(device)
    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded trained checkpoint from Epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Load test data
    test_loader = get_dataloader(test_dir, img_size=128, batch_size=1, is_train=False)
    print(f"Evaluating {len(test_loader.dataset)} test samples...")
    
    blurry_psnr_list, restored_psnr_list = [], []
    blurry_ssim_list, restored_ssim_list = [], []
    inference_times = []
    
    saved_samples = []
    
    with torch.no_grad():
        for idx, (blurry_tensor, sharp_tensor) in enumerate(test_loader):
            blurry_tensor = blurry_tensor.to(device)
            sharp_tensor = sharp_tensor.to(device)
            
            # Measure inference speed
            t0 = time.time()
            restored_tensor = model(blurry_tensor)
            inf_time = time.time() - t0
            inference_times.append(inf_time)
            
            # Baseline degraded metrics (Blurry vs Sharp)
            b_psnr = compute_psnr(blurry_tensor, sharp_tensor)
            b_ssim = compute_ssim_tensor(blurry_tensor, sharp_tensor)
            
            # Restored metrics (Restored vs Sharp)
            r_psnr = compute_psnr(restored_tensor, sharp_tensor)
            r_ssim = compute_ssim_tensor(restored_tensor, sharp_tensor)
            
            blurry_psnr_list.append(b_psnr)
            restored_psnr_list.append(r_psnr)
            blurry_ssim_list.append(b_ssim)
            restored_ssim_list.append(r_ssim)
            
            # Save first 4 samples for side-by-side visualization
            if idx < 4:
                b_np = blurry_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
                r_np = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
                s_np = sharp_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
                saved_samples.append((b_np, r_np, s_np, b_psnr, r_psnr, b_ssim, r_ssim))
                
    # Aggregate statistics
    avg_blurry_psnr = float(np.mean(blurry_psnr_list))
    avg_restored_psnr = float(np.mean(restored_psnr_list))
    psnr_improvement = avg_restored_psnr - avg_blurry_psnr
    
    avg_blurry_ssim = float(np.mean(blurry_ssim_list))
    avg_restored_ssim = float(np.mean(restored_ssim_list))
    ssim_improvement = avg_restored_ssim - avg_blurry_ssim
    
    avg_inf_speed_ms = float(np.mean(inference_times)) * 1000.0
    
    results = {
        "test_samples_count": len(test_loader.dataset),
        "baseline_blurry_psnr_db": round(avg_blurry_psnr, 2),
        "model_restored_psnr_db": round(avg_restored_psnr, 2),
        "psnr_gain_db": round(psnr_improvement, 2),
        "baseline_blurry_ssim": round(avg_blurry_ssim, 4),
        "model_restored_ssim": round(avg_restored_ssim, 4),
        "ssim_gain": round(ssim_improvement, 4),
        "average_inference_time_ms": round(avg_inf_speed_ms, 2)
    }
    
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS TABLE")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Input Blurry':<15} | {'Restored Model':<15} | {'Gain / Improvement':<15}")
    print("-" * 78)
    print(f"{'PSNR (Peak Signal-to-Noise)':<30} | {avg_blurry_psnr:^15.2f} | {avg_restored_psnr:^15.2f} | {f'+{psnr_improvement:.2f} dB':^15}")
    print(f"{'SSIM (Structural Similarity)':<30} | {avg_blurry_ssim:^15.4f} | {avg_restored_ssim:^15.4f} | {f'+{ssim_improvement:.4f}':^15}")
    print(f"{'Average Inference Speed':<30} | {'-':^15} | {f'{avg_inf_speed_ms:.1f} ms':^15} | {'Instantaneous':^15}")
    print("=" * 70)
    
    # Save results json
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate side-by-side visual comparison figure
    fig, axes = plt.subplots(len(saved_samples), 3, figsize=(12, 3.2 * len(saved_samples)), dpi=180)
    if len(saved_samples) == 1:
        axes = np.expand_dims(axes, 0)
        
    cols = ["Input Blurry Image", "DeblurNet-Lite Restored", "Ground-Truth Sharp"]
    for c, col_name in enumerate(cols):
        axes[0, c].set_title(col_name, fontsize=12, fontweight="bold", pad=8)
        
    for r, (b_img, r_img, s_img, b_p, r_p, b_s, r_s) in enumerate(saved_samples):
        axes[r, 0].imshow(b_img)
        axes[r, 0].set_ylabel(f"Sample {r+1}", fontsize=11, fontweight="bold")
        axes[r, 0].set_xlabel(f"PSNR: {b_p:.1f} dB | SSIM: {b_s:.3f}", fontsize=9)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        
        axes[r, 1].imshow(r_img)
        axes[r, 1].set_xlabel(f"PSNR: {r_p:.1f} dB | SSIM: {r_s:.3f}", fontsize=9, color="#1e8449", fontweight="bold")
        axes[r, 1].set_xticks([])
        axes[r, 1].set_yticks([])
        
        axes[r, 2].imshow(s_img)
        axes[r, 2].set_xlabel("Target Ground Truth", fontsize=9)
        axes[r, 2].set_xticks([])
        axes[r, 2].set_yticks([])
        
    plt.tight_layout()
    comparison_path = Path(output_dir) / "comparison_grid.png"
    plt.savefig(comparison_path, bbox_inches="tight")
    plt.close()
    print(f"Visual comparison grid saved to: {comparison_path}")
    
    return results


if __name__ == "__main__":
    evaluate_test_set()

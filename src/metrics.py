import sys
import math
from pathlib import Path
from typing import Tuple

site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

import torch
import numpy as np


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) between two image tensors.
    
    Formula:
      MSE = mean((pred - target)^2)
      PSNR = 10 * log10(max_val^2 / MSE)
      
    Typical ranges:
      - Blurry images: 18 - 23 dB
      - Restored images: 24 - 30+ dB
      - Identical images: infinity (capped at 100 dB)
    """
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0:
            return 100.0
        return 10.0 * math.log10((max_val ** 2) / mse)


def compute_ssim_tensor(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two 4D tensors (B, C, H, W).
    SSIM measures structural fidelity, luminance, and contrast preservation.
    Range: [-1.0, 1.0], where 1.0 represents perfect structural identity.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        # Convert tensors to numpy arrays: (H, W, C)
        p_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        t_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        
        # Calculate multi-channel SSIM
        score = ssim_fn(t_np, p_np, data_range=1.0, channel_axis=2)
        return float(score)
    except Exception:
        # Robust fallback using simplified Gaussian/patch-based formulation
        p_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        t_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        
        mu_x = p_np.mean()
        mu_y = t_np.mean()
        
        sigma_x_sq = np.var(p_np)
        sigma_y_sq = np.var(t_np)
        sigma_xy = np.mean((p_np - mu_x) * (t_np - mu_y))
        
        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
        return float(num / den)


if __name__ == "__main__":
    t1 = torch.rand(1, 3, 128, 128)
    t2 = t1 + torch.randn(1, 3, 128, 128) * 0.05
    t2 = torch.clamp(t2, 0.0, 1.0)
    
    psnr_val = compute_psnr(t2, t1)
    ssim_val = compute_ssim_tensor(t2, t1)
    print(f"Sample Metric Test -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

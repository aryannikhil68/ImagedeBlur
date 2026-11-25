import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np

site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from src.model import DeblurNetLite, count_parameters
from src.dataset import get_dataloader
from src.metrics import compute_psnr, compute_ssim_tensor


class CompoundDeblurLoss(nn.Module):
    """
    Compound Deblurring Loss: L1 Pixel Loss + Sobel Edge Gradient Loss.
    
    Why this is essential:
    - Pure L1 loss only matches average color intensities, often leaving edges slightly soft.
    - Sobel Edge Gradient Loss explicitly computes the difference between predicted edge gradients
      and true sharp edge gradients in both X and Y directions.
    - This mathematically forces the network to sharpen lines, window borders, and high-frequency textures!
    """
    def __init__(self, edge_weight: float = 0.5):
        super(CompoundDeblurLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_weight = edge_weight
        
        # 3x3 Sobel filters for edge detection
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3)
                                
        # Repeat across 3 RGB channels (depthwise convolution)
        self.register_buffer("kernel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("kernel_y", sobel_y.repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Pixel-level L1 loss
        loss_pixel = self.l1_loss(pred, target)
        
        # 2. High-frequency edge gradient loss
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1, groups=3)
        targ_grad_x = F.conv2d(target, self.kernel_x, padding=1, groups=3)
        targ_grad_y = F.conv2d(target, self.kernel_y, padding=1, groups=3)
        
        loss_edge = F.l1_loss(pred_grad_x, targ_grad_x) + F.l1_loss(pred_grad_y, targ_grad_y)
        
        return loss_pixel + self.edge_weight * loss_edge


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeblurNet-Lite Model with Global Residual Learning")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (optimized for 4GB RAM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--img_size", type=int, default=128, help="Image resolution (HxW)")
    parser.add_argument("--quick_test", action="store_true", help="Quick test run (2 epochs)")
    return parser.parse_args()


def plot_training_curves(history, output_dir="results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs_range, history["train_loss"], label="Train Compound Loss", color="#e74c3c", lw=2, marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Compound Loss", color="#2980b9", lw=2, marker='s')
    plt.title("DeblurNet-Lite: Training & Validation Loss (L1 + Sobel Edge Loss)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Compound Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(Path(output_dir) / "loss_curve.png", bbox_inches="tight")
    plt.close()
    
    # 2. PSNR Curve
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs_range, history["val_psnr"], label="Validation PSNR (dB)", color="#27ae60", lw=2, marker='^')
    plt.title("DeblurNet-Lite: Validation PSNR Restoration Improvement", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("PSNR (dB)", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(Path(output_dir) / "psnr_curve.png", bbox_inches="tight")
    plt.close()


def train():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        print(f"Hardware Mode: Running on CPU (Threads limited to {torch.get_num_threads()} for stability)")
    else:
        print(f"Hardware Mode: Running on GPU ({torch.cuda.get_device_name(0)})")
        
    ckpt_dir = Path("checkpoints")
    results_dir = Path("results")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = 2 if args.quick_test else args.epochs
    
    train_loader = get_dataloader("data/train", img_size=args.img_size, batch_size=args.batch_size, is_train=True)
    val_loader = get_dataloader("data/val", img_size=args.img_size, batch_size=args.batch_size, is_train=False)
    
    print(f"Loaded: {len(train_loader.dataset)} training samples | {len(val_loader.dataset)} validation samples")
    
    model = DeblurNetLite().to(device)
    print(f"Model: DeblurNetLite (Global Residual) initialized ({count_parameters(model):,} parameters).")
    
    # Compound Loss with Sobel Edge Sharpening
    criterion = CompoundDeblurLoss(edge_weight=0.5).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}
    best_val_loss = float("inf")
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Val PSNR (dB)':^14} | {'Val SSIM':^10} | {'Time (s)':^8}")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for blurry_imgs, sharp_imgs in train_loader:
            blurry_imgs = blurry_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(blurry_imgs)
            loss = criterion(outputs, sharp_imgs)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * blurry_imgs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_psnr_list = []
        val_ssim_list = []
        
        with torch.no_grad():
            for blurry_imgs, sharp_imgs in val_loader:
                blurry_imgs = blurry_imgs.to(device)
                sharp_imgs = sharp_imgs.to(device)
                
                outputs = model(blurry_imgs)
                loss = criterion(outputs, sharp_imgs)
                running_val_loss += loss.item() * blurry_imgs.size(0)
                
                for b in range(blurry_imgs.size(0)):
                    psnr = compute_psnr(outputs[b:b+1], sharp_imgs[b:b+1])
                    ssim = compute_ssim_tensor(outputs[b:b+1], sharp_imgs[b:b+1])
                    val_psnr_list.append(psnr)
                    val_ssim_list.append(ssim)
                    
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_psnr = float(np.mean(val_psnr_list))
        epoch_val_ssim = float(np.mean(val_ssim_list))
        epoch_duration = time.time() - epoch_start
        
        scheduler.step(epoch_val_loss)
        
        history["train_loss"].append(round(epoch_train_loss, 5))
        history["val_loss"].append(round(epoch_val_loss, 5))
        history["val_psnr"].append(round(epoch_val_psnr, 2))
        history["val_ssim"].append(round(epoch_val_ssim, 4))
        
        print(f"{epoch:^7d} | {epoch_train_loss:^12.5f} | {epoch_val_loss:^12.5f} | {epoch_val_psnr:^14.2f} | {epoch_val_ssim:^10.4f} | {epoch_duration:^8.1f}")
        
        # Save Best Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": epoch_val_loss,
                "val_psnr": epoch_val_psnr,
                "val_ssim": epoch_val_ssim
            }, ckpt_dir / "best_model.pth")
            
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": epoch_val_loss
        }, ckpt_dir / "latest_model.pth")
        
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_time / 60:.2f} minutes!")
    print(f"Best Validation Loss: {best_val_loss:.5f}")
    print(f"Model checkpoint saved at: {ckpt_dir / 'best_model.pth'}")
    
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    plot_training_curves(history, output_dir=str(results_dir))


if __name__ == "__main__":
    train()

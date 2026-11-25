import os
import sys
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

def create_motion_blur_kernel(size=7, angle_degrees=45):
    """
    Creates a realistic 2D motion blur kernel (Point Spread Function - PSF).
    Simulates linear camera shake during exposure.
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radian = math.radians(angle_degrees)
    cos_val = math.cos(radian)
    sin_val = math.sin(radian)
    
    for i in range(-center, center + 1):
        x = int(round(center + i * cos_val))
        y = int(round(center + i * sin_val))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
            
    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0
    return kernel

def apply_custom_motion_blur(img: Image.Image, kernel: np.ndarray) -> Image.Image:
    """Applies 2D PSF convolution to an RGB image."""
    from scipy.signal import convolve2d
    
    img_arr = np.array(img, dtype=np.float32)
    blurred_arr = np.zeros_like(img_arr)
    
    for c in range(3):
        blurred_arr[:, :, c] = convolve2d(img_arr[:, :, c], kernel, mode='same', boundary='symm')
        
    blurred_arr = np.clip(blurred_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred_arr)

def generate_scene_128(seed_val: int) -> Image.Image:
    """
    Generates structured scenes directly at 128x128 pixels.
    Contains high-contrast geometric objects, architectural lines, windows, and fine details.
    """
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    width, height = 128, 128
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 1. Background Sky wash
    top_color = (random.randint(40, 140), random.randint(70, 160), random.randint(180, 245))
    bottom_color = (random.randint(180, 230), random.randint(200, 240), random.randint(220, 255))
    for y in range(height // 2):
        r_val = y / (height // 2)
        r = int(top_color[0] * (1 - r_val) + bottom_color[0] * r_val)
        g = int(top_color[1] * (1 - r_val) + bottom_color[1] * r_val)
        b = int(top_color[2] * (1 - r_val) + bottom_color[2] * r_val)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    # 2. Road/Ground
    draw.rectangle([0, int(height * 0.7), width, height], fill=(70, 75, 80))
    for rx in range(6, width, 18):
        draw.rectangle([rx, int(height * 0.84), rx + 10, int(height * 0.84) + 2], fill=(255, 255, 255))
        
    # 3. Buildings with distinct high-frequency window grids
    num_buildings = random.randint(3, 4)
    for b in range(num_buildings):
        bx = random.randint(0, width - 35)
        bw = random.randint(26, 45)
        bh = random.randint(45, 80)
        by = int(height * 0.7) - bh
        b_color = (random.randint(80, 210), random.randint(80, 210), random.randint(80, 210))
        draw.rectangle([bx, by, bx + bw, int(height * 0.7)], fill=b_color, outline=(20, 20, 20), width=1)
        
        # High-contrast windows (essential for learning edge restoration)
        for wy in range(by + 6, int(height * 0.7) - 8, 10):
            for wx in range(bx + 4, bx + bw - 6, 8):
                win_fill = (255, 255, 180) if random.random() > 0.4 else (25, 35, 45)
                draw.rectangle([wx, wy, wx + 5, wy + 6], fill=win_fill, outline=(10, 10, 10), width=1)
                
        # Signboard or accent strip
        if random.random() > 0.3:
            draw.rectangle([bx + 2, by + 1, bx + bw - 2, by + 5], fill=(220, 30, 30))
            
    # 4. Fine high-contrast utility lines across the sky
    for _ in range(random.randint(2, 3)):
        y1 = random.randint(15, 50)
        y2 = random.randint(15, 50)
        draw.line([(0, y1), (width, y2)], fill=(15, 15, 15), width=1)
        
    # 5. Sun / Circular element
    cx = random.randint(15, width - 20)
    cy = random.randint(12, 40)
    cr = random.randint(7, 12)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(255, 220, 50), outline=(40, 40, 40), width=1)
    
    return img

def prepare_dataset(base_dir="data", total_samples=300):
    print(f"Generating native 128x128 paired dataset in '{base_dir}'...")
    base_path = Path(base_dir)
    
    splits = {
        "train": (0, int(total_samples * 0.8)),
        "val": (int(total_samples * 0.8), int(total_samples * 0.9)),
        "test": (int(total_samples * 0.9), total_samples)
    }
    
    for split_name, (start_idx, end_idx) in splits.items():
        blurry_dir = base_path / split_name / "blurry"
        sharp_dir = base_path / split_name / "sharp"
        blurry_dir.mkdir(parents=True, exist_ok=True)
        sharp_dir.mkdir(parents=True, exist_ok=True)
        
        count = end_idx - start_idx
        print(f"Generating {split_name.upper()} set: {count} paired scenes (Native 128x128)...")
        
        for idx in range(start_idx, end_idx):
            sharp_img = generate_scene_128(seed_val=3000 + idx)
            
            # Motion blur kernel size 5 to 9 pixels (realistic for 128x128 image)
            blur_kernel_size = random.choice([5, 7, 9])
            blur_angle = random.randint(0, 180)
            kernel = create_motion_blur_kernel(size=blur_kernel_size, angle_degrees=blur_angle)
            
            blurry_img = apply_custom_motion_blur(sharp_img, kernel)
            
            filename = f"pair_{idx:04d}.png"
            sharp_img.save(sharp_dir / filename)
            blurry_img.save(blurry_dir / filename)
            
    print(f"\nSUCCESS: Native 128x128 dataset generated in '{base_dir}'!")

if __name__ == "__main__":
    prepare_dataset(base_dir="data", total_samples=300)

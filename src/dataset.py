import os
import sys
import random
from pathlib import Path
from typing import Tuple, Optional

site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class DeblurDataset(Dataset):
    """
    Custom PyTorch Dataset for Paired Blurry-Sharp Image Supervised Deblurring.
    
    Structure required:
      split_dir /
        ├── blurry /  (e.g., pair_0001.png)
        └── sharp /   (e.g., pair_0001.png)
        
    Guarantees:
    - Pixel-perfect spatial alignment between blurry and sharp images.
    - Synchronized data augmentation (flips applied jointly to both).
    - Bicubic resizing to manageable dimensions (default: 128x128).
    - Normalization of pixel intensities to [0.0, 1.0].
    """
    def __init__(self, split_dir: str, img_size: int = 128, is_train: bool = False):
        super(DeblurDataset, self).__init__()
        self.split_dir = Path(split_dir)
        self.blurry_dir = self.split_dir / "blurry"
        self.sharp_dir = self.split_dir / "sharp"
        self.img_size = (img_size, img_size)
        self.is_train = is_train
        
        if not self.blurry_dir.exists() or not self.sharp_dir.exists():
            raise FileNotFoundError(f"Missing blurry or sharp subdirectories in: {split_dir}")
            
        # Collect and match filenames
        valid_extensions = {".png", ".jpg", ".jpeg"}
        blurry_files = sorted([f.name for f in self.blurry_dir.iterdir() if f.suffix.lower() in valid_extensions])
        sharp_files = set([f.name for f in self.sharp_dir.iterdir() if f.suffix.lower() in valid_extensions])
        
        # Verify strict 1-to-1 pairing
        self.filenames = [f for f in blurry_files if f in sharp_files]
        if len(self.filenames) == 0:
            raise ValueError(f"No matching blurry-sharp image pairs found in: {split_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        blurry_path = self.blurry_dir / fname
        sharp_path = self.sharp_dir / fname
        
        blurry_img = Image.open(blurry_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")
        
        # 1. Resize to target dimensions
        blurry_img = TF.resize(blurry_img, self.img_size, interpolation=TF.InterpolationMode.BICUBIC)
        sharp_img = TF.resize(sharp_img, self.img_size, interpolation=TF.InterpolationMode.BICUBIC)
        
        # 2. Synchronized Data Augmentation (Train set only)
        # Flipping must be applied identically to both images to maintain pixel alignment!
        if self.is_train:
            if random.random() > 0.5:
                blurry_img = TF.hflip(blurry_img)
                sharp_img = TF.hflip(sharp_img)
            if random.random() > 0.5:
                blurry_img = TF.vflip(blurry_img)
                sharp_img = TF.vflip(sharp_img)
                
        # 3. Convert PIL Image to PyTorch Tensor [0.0, 1.0]
        blurry_tensor = TF.to_tensor(blurry_img)
        sharp_tensor = TF.to_tensor(sharp_img)
        
        return blurry_tensor, sharp_tensor


def get_dataloader(
    split_dir: str,
    img_size: int = 128,
    batch_size: int = 4,
    is_train: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates an optimized PyTorch DataLoader suitable for CPU execution.
    - batch_size=4: Low RAM consumption.
    - num_workers=0: Eliminates Windows multiprocessing overhead and memory thrashing.
    """
    dataset = DeblurDataset(split_dir=split_dir, img_size=img_size, is_train=is_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=False
    )
    return loader


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DeblurDataset & DataLoader...")
    print("=" * 60)
    
    test_dir = Path("data/train")
    if not test_dir.exists():
        print("Note: Run 'download_or_create_data.py' first to generate dataset!")
    else:
        loader = get_dataloader(str(test_dir), img_size=128, batch_size=4, is_train=True)
        print(f"DataLoader created with {len(loader)} batches ({len(loader.dataset)} total samples).")
        
        blurry_batch, sharp_batch = next(iter(loader))
        print(f"Blurry Batch Shape: {blurry_batch.shape} | Range: [{blurry_batch.min():.2f}, {blurry_batch.max():.2f}]")
        print(f"Sharp Batch Shape:  {sharp_batch.shape} | Range: [{sharp_batch.min():.2f}, {sharp_batch.max():.2f}]")
        print("SUCCESS: Dataset loader passed all validation tests!")
        print("=" * 60)

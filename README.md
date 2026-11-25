# AI-Based Image Deblurring Using a Custom Deep Learning Model

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU--Friendly%20(4GB%20RAM)-green.svg)]()
[![Model](https://img.shields.io/badge/Model-DeblurNet--Lite%20(293K%20Params)-brightgreen.svg)]()

> **Final Year College Academic Group Project**  
> **Title:** AI-Based Image Deblurring Using a Custom Deep Learning Model  
> **Team:** 3 Students (Architect, ML Engineer, Deployment Lead)  
> **Status:** Fully functional, demonstrable, and evaluated from scratch without pretrained weights.

---

## 1. Project Introduction

In everyday photography, mobile imaging, medical diagnostics, and traffic surveillance, image blur caused by hand tremors, moving objects, or lens defocus is one of the most common forms of image degradation. 

Traditional image restoration techniques (such as Wiener filtering and Richardson-Lucy deconvolution) require precise mathematical knowledge of the blur trajectory (the blur kernel or Point Spread Function), which is almost impossible to determine in real life. Furthermore, they are notoriously sensitive to sensor noise and produce severe ringing artifacts.

This project introduces **DeblurNet-Lite**, an original, lightweight convolutional neural network with symmetric skip connections and residual bottleneck blocks. The model was designed, implemented, and trained **100% from scratch** on a standard laptop with **4 GB RAM and no dedicated GPU**.

---

## 2. Problem Statement

Commercial state-of-the-art deblurring models (such as Restormer, NAFNet, and DeblurGAN-v2) have 30M to 100M parameters and demand enterprise-grade GPUs (16GB+ VRAM). For students, researchers with limited hardware, and edge computing devices, these models are completely inaccessible and crash standard laptops with Out-Of-Memory (OOM) errors.

Furthermore, many academic projects simply download black-box pretrained weights, preventing students from understanding the fundamental mechanics of how neural networks learn feature extraction, residual correction, and edge reconstruction.

**Our Core Challenge:**
To design an original, lightweight neural network architecture (< 300,000 parameters), train it from randomly initialized weights on local CPU hardware, achieve statistically significant quantitative gains in Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), and deliver a functional, interactive web application.

---

## 3. Project Objectives

1. **Design a Custom Architecture:** Implement `DeblurNet-Lite`, an original compact residual convolutional autoencoder (~293K parameters).
2. **Leak-Free Dataset Pipeline:** Create an automated, strictly partitioned paired dataset (80% train, 10% val, 10% test) with zero data leakage.
3. **Train Completely from Scratch:** Initialize with Kaiming (He) Normal weights and optimize on local CPU using L1 restoration loss and the Adam optimizer.
4. **Rigorous Scientific Evaluation:** Benchmark performance using standard scientific metrics: PSNR (dB), SSIM, and CPU inference latency.
5. **Interactive Web Application:** Build a responsive Streamlit frontend allowing real-time image upload, instant restoration, side-by-side comparison, and high-resolution export.
6. **Cloud Readiness:** Structure the codebase for zero-cost deployment on Streamlit Community Cloud.

---

## 4. Technologies & Tools Used

- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch & Torchvision (CPU optimized)
- **Web Application Framework:** Streamlit
- **Computer Vision & Scientific Stack:** NumPy, SciPy, Pillow, Scikit-Image, Matplotlib
- **Presentation & Documentation:** Python-PPTX (12-slide presentation), ReportLab (PDF Report)
- **Deployment Platform:** Streamlit Community Cloud & GitHub

---

## 5. System Architecture: DeblurNet-Lite

DeblurNet-Lite operates as an end-to-end regression autoencoder containing exactly **293,642 trainable parameters**:

```
[Input Blurry Image] (B, 3, 128, 128)
         │
         ▼
[Encoder Conv 1] (3 -> 32 channels, 3x3 kernel, stride 1, PReLU) ──┐ (Symmetric Skip Connection)
         │                                                        │
         ▼                                                        │
[Downsampling Conv] (32 -> 64 channels, 3x3 kernel, stride 2, BN, PReLU)
         │
         ▼
[Residual Bottleneck] (3x Residual Blocks: Output = x + F(x))
         │
         ▼
[Decoder ConvTranspose] (64 -> 32 channels, 4x4 kernel, stride 2, BN, PReLU)
         │
         ▼
[Skip Fusion] (Concatenate: 32 + 32 = 64 channels -> Conv 3x3 -> 32 channels) ◄──┘
         │
         ▼
[Reconstruction Layer] (Conv 3x3, 32 -> 3 channels + Sigmoid Activation)
         │
         ▼
[Output Sharp Image] (B, 3, 128, 128) bounded strictly to [0.0, 1.0]
```

### Why this architecture?
- **Residual Blocks ($x + F(x)$):** Enable smooth backward gradient flow, preventing gradient vanishing and allowing the network to learn subtle edge deltas.
- **Symmetric Skip Connection:** Bypasses downsampling to inject high-resolution spatial boundary information directly into the decoder.
- **L1 Loss Optimization:** Penalizes absolute errors linearly, avoiding the blurry, over-smoothed edges caused by standard L2 (MSE) loss.
- **Sigmoid Activation:** Mathematically guarantees reconstructed pixel intensities stay bounded between 0.0 and 1.0.

---

## 6. Dataset and Preprocessing Pipeline

- **Total Samples:** 300 strictly paired image scenes.
- **Partitioning:**
  - `data/train/`: 240 pairs (80%)
  - `data/val/`: 30 pairs (10%)
  - `data/test/`: 30 pairs (10%)
- **Zero Data Leakage:** Scenes are assigned unique IDs and partitioned *prior* to blur convolution.
- **Degradation Mechanics:** 2D Point Spread Function (PSF) motion blur kernels (lengths 7–13 px, angles 0°–180°) plus subtle optical defocus.
- **Preprocessing:**
  - Bicubic resizing to $128 \times 128$ pixels.
  - Normalization to $[0.0, 1.0]$ floating-point tensors.
  - Synchronized horizontal and vertical flips applied identically to both images.

---

## 7. Installation and Quickstart

### Prerequisites
Make sure Python 3.10 or higher is installed on your system.

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/ai-image-deblurring.git
cd ai-image-deblurring
pip install -r requirements.txt
```

### Step 2: Generate Dataset
```bash
python download_or_create_data.py
```

### Step 3: Train Model from Scratch (CPU)
```bash
# Full training (15 epochs - ~6 minutes on CPU)
python train.py --epochs 15 --batch_size 4

# Or quick verification run (2 epochs - ~45 seconds)
python train.py --quick_test
```

### Step 4: Evaluate on Independent Test Set
```bash
python evaluate.py
```

### Step 5: Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 8. Experimental Results

Evaluated on **30 unseen test images**:

| Metric | Input Blurry Image | DeblurNet-Lite Restored | Net Gain / Improvement |
| :--- | :---: | :---: | :---: |
| **PSNR (Peak Signal-to-Noise)** | **19.71 dB** | **25.72 dB** | **+6.02 dB** |
| **SSIM (Structural Similarity)** | **0.6934** | **0.9331** | **+0.2397 (+34.5%)** |
| **Inference Latency (CPU)** | — | **~23.1 ms / image** | **Near-Real-Time (43 FPS)** |
| **Memory Footprint (RAM)** | — | **< 380 MB** | **Zero OOM Crashes** |

*Generated artifacts:*
- Architecture Diagram: `results/architecture_diagram.png`
- Loss Convergence Curves: `results/loss_curve.png`
- Validation PSNR Curves: `results/psnr_curve.png`
- Visual Comparison Grid: `results/comparison_grid.png`

---

## 9. Team Responsibilities & Presentation

- **Student 1 (Architect):** Theoretical Foundations, Degradation Physics, Model Architecture (Slides 1–4).
- **Student 2 (ML Engineer):** Dataset Pipeline, Preprocessing, Supervised Training, L1 Loss (Slides 5–8).
- **Student 3 (Deployment Lead):** Evaluation Metrics, Visual Comparison, Streamlit Frontend, Viva Defense (Slides 9–12).

*Supporting Documents:*
- Presentation Slides: `AI_Image_Deblurring_Presentation.pptx` (Exactly 12 slides)
- Word-for-Word Spoken Script: `Presentation_Script.txt`
- 65+ Professor Viva Q&A: `Project_Viva_Questions.txt`
- Group Allocation Matrix: `Group_Task_Allocation.md`
- Cloud Deployment Guide: `DEPLOYMENT_GUIDE.md`

---

## 10. Honest Limitations & Future Scope

### Limitations
1. **Severe Motion Blur:** For extreme blur (> 25 pixel displacement), the compact receptive field produces smooth restorations rather than razor-sharp textures.
2. **Patch Resizing:** High-resolution photos are scaled down to 256x256 before inference to preserve CPU responsiveness.
3. **Single Image:** Operates on static images; does not leverage temporal coherence across video frames.

### Future Scope
1. **Perceptual Loss:** Integrating VGG-19 feature space loss for richer high-frequency textures.
2. **Channel Attention:** Adding lightweight Squeeze-and-Excitation blocks.
3. **Video Deblurring:** Incorporating sequential temporal frame warping.
4. **ONNX Export:** Quantizing model for mobile and edge device deployment.

---

## 11. Academic License & Attribution

This project is submitted for academic credit. Released under the MIT License.

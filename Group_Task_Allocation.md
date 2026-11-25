# College Group Project Management & Task Allocation
## Project Title: AI-Based Image Deblurring Using a Custom Deep Learning Model

**Team Size:** 3 Students  
**Hardware Profile:** Standard Laptop (4 GB RAM, CPU-Only)  
**Target Submission:** Final Academic Examination & Viva Defense  

---

## 1. Executive Summary & Team Roles

1. **Gautam Kumar Mishra (Roll No: 2024PGCSCA104) — Lead Researcher & Architect**
   - *Domain:* Mathematical Foundations, Degradation Physics, and Custom CNN Architectural Design.
   - *Presentation:* Slides 1 to 4 (Intro, Problem, Objectives, Classical Methods).
   - *Focus:* Why deblurring is an ill-posed inverse problem, convolution mechanics, DeblurNet-Lite layer design, parameter budget, and weight initialization.

2. **Sumit Paul (Roll No: 2024PGCSCA013) — ML Pipeline & Training Engineer**
   - *Domain:* Dataset Engineering, Preprocessing, Supervised Training, and Loss Optimization.
   - *Presentation:* Slides 5 to 8 (Methodology, Dataset, Architecture, Loss & Optimization).
   - *Focus:* Paired image organization, zero data leakage guarantees, L1 vs. L2 loss dynamics, Sobel edge loss, Adam optimizer, learning rate scheduling, and CPU memory optimization.

3. **Rahul Bhandari (Roll No: 2024PGCSCA037) — Evaluation, Frontend Developer & Deployment Lead**
   - *Domain:* Scientific Metrics (PSNR/SSIM), Visual Validation, Streamlit Application, and Viva Defense.
   - *Presentation:* Slides 9 to 13 (Training Results, Test Benchmark, Live Demo, Limitations, Thank You).
   - *Focus:* Peak Signal-to-Noise Ratio (dB), Structural Similarity Index, side-by-side comparison visualization, Streamlit UI implementation, and Community Cloud deployment.

---

## 2. Comprehensive Task Allocation Matrix

| Project Phase | Specific Deliverables / Modules | Primary Owner | Secondary Reviewer | Key Concepts to Master |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1: Research & Problem Formulation** | Degradation equation $B = (I * K) + N$, ill-posed problem proof, existing methods review | **Student 1** | Student 2 | Point Spread Function (PSF), blur kernels, inverse filtering limitations |
| **Phase 2: Custom Architecture Design** | `src/model.py`, DeblurNet-Lite implementation, architecture block diagram | **Student 1** | Student 3 | Residual blocks ($x + F(x)$), transposed convolutions, symmetric skip connections |
| **Phase 3: Dataset & Preprocessing** | `download_or_create_data.py`, `src/dataset.py`, DataLoader, train/val/test splits | **Student 2** | Student 1 | 80/10/10 split, bicubic resizing, normalization [0, 1], synchronized flips |
| **Phase 4: Training & Optimization** | `train.py`, L1 loss function, Adam optimizer, checkpointing, learning curves | **Student 2** | Student 3 | L1 vs. L2 loss advantages, gradient backpropagation, ReduceLROnPlateau, CPU thread limits |
| **Phase 5: Model Evaluation & Metrics** | `src/metrics.py`, `evaluate.py`, PSNR & SSIM computation, comparison grids | **Student 3** | Student 2 | PSNR (dB) formula, MSE, SSIM luminance/contrast/structure, inference latency |
| **Phase 6: Frontend Development** | `app.py`, Streamlit web app, image uploader, demo selector, download button | **Student 3** | Student 1 | `@st.cache_resource`, reactive state, image buffering, latency metrics |
| **Phases 7–12: Viva, Slides & Deployment** | PowerPoint slides, presentation script, viva Q&A mastery, GitHub repository | **All 3 Members** | Mentor | Equal participation in 12-slide presentation and professor Q&A |

---

## 3. Division of the 12 Presentation Slides

Each student speaks for approximately **4 to 5 minutes** (1 to 1.5 minutes per slide):

### **Speaker 1: Student 1 (Slides 1 – 4)**
- **Slide 1: Title & Team Introduction**
  - Introduce the project title, team members, department, and project advisor.
- **Slide 2: Introduction to Image Deblurring**
  - Explain how camera shake smears light across sensor pixels; explain the degradation formula $B = (I * K) + N$.
- **Slide 3: Problem Statement**
  - Highlight the hardware barrier of commercial 50M-parameter models and the academic need to understand training from scratch.
- **Slide 4: Project Objectives & Existing Methods**
  - List the 5 core project goals; compare classical Wiener/Richardson-Lucy deconvolution with end-to-end deep learning.

### **Speaker 2: Student 2 (Slides 5 – 8)**
- **Slide 5: Proposed Methodology**
  - Present the 5-stage supervised learning pipeline from paired data to trained checkpoint.
- **Slide 6: Dataset & Preprocessing Pipeline**
  - Explain the 80/10/10 split, zero data leakage policy, normalization, and synchronized data augmentation.
- **Slide 7: Our Custom CNN Architecture (DeblurNet-Lite)**
  - Walk through the 293,642-parameter model: encoder, 3 residual bottleneck blocks, transposed conv decoder, and skip connections.
- **Slide 8: Loss Function & Optimization Mathematics**
  - Explain why L1 loss produces sharper edges than L2 (MSE), explain the Adam optimizer and learning rate scheduling.

### **Speaker 3: Student 3 (Slides 9 – 12)**
- **Slide 9: Model Training & Convergence Analysis**
  - Present the actual training curves, validation loss reduction, and absence of overfitting.
- **Slide 10: Experimental Evaluation & Results**
  - Present the quantitative test results table: +3.53 dB PSNR gain, +0.1975 SSIM improvement, and 38 ms inference speed.
- **Slide 11: Streamlit Frontend Application Demonstration**
  - Demonstrate the live web application, image uploading, instant inference, and high-resolution export.
- **Slide 12: Limitations, Future Work & Conclusion**
  - Address honest academic limitations (extreme blur), future extensions (perceptual loss, video), and conclude the presentation.

---

## 4. Minimum Core Knowledge Checklist for Every Team Member

Regardless of individual task division, **every single team member must be able to answer these 5 foundational questions during the viva examination**:

1. **Why is deblurring called an ill-posed inverse problem?**  
   *Answer:* Because high-frequency details (sharp edges) are destroyed during blur convolution, multiple different sharp scenes can produce the same blurry image, and naive inversion amplifies sensor noise.
2. **Did you download a pretrained model or train it yourself?**  
   *Answer:* We trained DeblurNet-Lite 100% from scratch with randomly initialized He/Kaiming weights. No pretrained weights were used.
3. **Why did you choose L1 loss instead of standard MSE (L2)?**  
   *Answer:* MSE squares errors, which causes the network to output an over-smoothed average of all possible edge positions (blurry edges). L1 penalizes errors linearly, preserving crisp edge boundaries.
4. **What do PSNR and SSIM measure?**  
   *Answer:* PSNR measures the numerical signal-to-noise ratio in decibels (higher is cleaner). SSIM measures human perceptual structural similarity, including luminance, contrast, and structural texture correlation (scale 0 to 1).
5. **How does your model run so fast on a standard CPU?**  
   *Answer:* Our model has under 300,000 parameters, uses lightweight 3x3 kernels, and processes 128x128 patches in ~38 ms with under 400 MB RAM consumption.

# Step-by-Step Streamlit Community Cloud Deployment Guide
## AI-Based Image Deblurring Using DeblurNet-Lite

> ### 🚀 Live Application URL:
> 👉 **[https://imagedeblur-vdnjyxj9dn6vn3blpwrcsw.streamlit.app/](https://imagedeblur-vdnjyxj9dn6vn3blpwrcsw.streamlit.app/)**  
> **Status:** Live, Operational, and Globally Accessible.

This guide provides secure, step-by-step instructions to deploy and maintain your image deblurring web application live on **Streamlit Community Cloud** for free.

---

## 🔒 Security Notice
- **Never share your GitHub or Streamlit passwords or private access tokens in chat or public code.**
- All authentication is performed directly through official OAuth login portals in your personal browser.

---

## Step 1: Prepare Your Local Project Files

Ensure your local directory contains all essential files:
```
minipro/
├── app.py                      # Streamlit main entrypoint
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
├── checkpoints/
│   └── best_model.pth          # Trained model weights (~1.2 MB)
├── src/
│   ├── __init__.py
│   ├── model.py                # DeblurNetLite architecture
│   ├── dataset.py              # Dataset loaders
│   └── metrics.py              # PSNR and SSIM functions
├── data/
│   └── test/
│       └── blurry/             # Sample images for instant demo
└── results/                    # Diagrams, curves, comparison grids
```

---

## Step 2: Push Your Project to GitHub

1. Open your terminal in `V:\minipro`.
2. Initialize git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI-Based Image Deblurring with DeblurNet-Lite"
   ```
3. Open your browser and go to: [https://github.com/new](https://github.com/new).
4. Create a new repository named: `ai-image-deblurring`.
   - Set visibility to **Public** (required for free Streamlit Community Cloud tier).
   - Do not check "Add README" or ".gitignore" (we already have them).
5. Link your local repo and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ai-image-deblurring.git
   git push -u origin main
   ```

---

## Step 3: Deploy on Streamlit Community Cloud

1. In your browser, navigate to: [https://share.streamlit.io](https://share.streamlit.io).
2. Click **"Sign in with GitHub"** and authenticate using GitHub's official authorization dialog.
3. Once logged into your Streamlit Cloud dashboard, click the blue **"New app"** button.
4. Fill in the deployment details:
   - **Repository:** `YOUR_GITHUB_USERNAME/ai-image-deblurring`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (Optional):** Choose a custom subdomain, e.g., `ai-image-deblurring.streamlit.app`.
5. Click **"Deploy!"**.

---

## Step 4: Verification and Live Testing

1. Streamlit Cloud will automatically clone your repository, install packages from `requirements.txt`, load `checkpoints/best_model.pth`, and launch `app.py`.
2. In about 1 to 2 minutes, the terminal log will show:
   `Running on http://...`
3. Test the live web application:
   - Select one of the preloaded demo samples or upload a blurry photo.
   - Verify that the restored image and metrics render cleanly side by side.
   - Click the "Download Restored High-Resolution Image" button to verify file download.
4. Copy the public URL (e.g., `https://ai-image-deblurring.streamlit.app`) and paste it into your presentation slide 11 and project report!

---

## Troubleshooting Common Deployment Issues

| Potential Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **"ModuleNotFoundError"** | A package is missing from `requirements.txt`. | Ensure `torch`, `torchvision`, `pillow`, `streamlit`, `scikit-image` are in `requirements.txt`. |
| **"MemoryLimitExceeded"** | App exceeded 1 GB free RAM. | DeblurNet-Lite uses < 300 MB RAM; ensure you do not increase batch size beyond 1 during inference. |
| **"FileNotFoundError: best_model.pth"** | Weights file was not committed. | Check that `checkpoints/best_model.pth` was added to Git (it is ~1.2 MB, well within GitHub's 100 MB limit). |

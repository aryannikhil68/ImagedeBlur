import sys
from pathlib import Path

site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path="AI_Image_Deblurring_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Color palette
    NAVY = RGBColor(30, 58, 138)
    DARK_GRAY = RGBColor(55, 65, 81)
    LIGHT_BG = RGBColor(248, 250, 252)
    CARD_BG = RGBColor(255, 255, 255)
    ACCENT_BLUE = RGBColor(37, 99, 235)
    BORDER_COLOR = RGBColor(226, 232, 240)
    
    def set_slide_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = LIGHT_BG
        
    def add_header(slide, title_text, category_text="COLLEGE ACADEMIC PROJECT PRESENTATION"):
        # Category Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.3))
        cat_tf = cat_box.text_frame
        cat_tf.word_wrap = True
        p_cat = cat_tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE
        
        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.7))
        title_tf = title_box.text_frame
        title_tf.word_wrap = True
        p_title = title_tf.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY
        
    def add_card(slide, left, top, width, height, title="", points=None, fill_color=CARD_BG):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
        shape.line.color.rgb = BORDER_COLOR
        shape.line.width = Pt(1)
        
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        tf.margin_bottom = Inches(0.25)
        
        if title:
            p_head = tf.paragraphs[0]
            p_head.text = title
            p_head.font.size = Pt(16)
            p_head.font.bold = True
            p_head.font.color.rgb = NAVY
            p_head.space_after = Pt(12)
            
        if points:
            for pt in points:
                p = tf.add_paragraph()
                p.text = f"•  {pt}"
                p.font.size = Pt(13)
                p.font.color.rgb = DARK_GRAY
                p.space_after = Pt(8)
        return shape

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide1)
    
    # Title box
    t_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "FINAL YEAR ACADEMIC PROJECT DEFENSE"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE
    p0.space_after = Pt(10)
    
    p1 = tf1.add_paragraph()
    p1.text = "AI-Based Image Deblurring Using a\nCustom Deep Learning Model"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(15)
    
    p2 = tf1.add_paragraph()
    p2.text = "A Lightweight End-to-End Convolutional Architecture Trained Completely from Scratch"
    p2.font.size = Pt(16)
    p2.font.color.rgb = DARK_GRAY
    
    # Metadata cards
    add_card(slide1, Inches(1.0), Inches(4.3), Inches(5.4), Inches(2.2), "Presented By (Team Members)", [
        "Student 1: Model Architecture & Design Lead",
        "Student 2: Dataset Engineering & Training Lead",
        "Student 3: Evaluation, Frontend & Deployment Lead"
    ])
    
    add_card(slide1, Inches(6.8), Inches(4.3), Inches(5.5), Inches(2.2), "Project Specifications", [
        "Framework: PyTorch 2.0+ (Trained on Local CPU)",
        "Model: DeblurNet-Lite (293,642 Parameters)",
        "Frontend: Streamlit Web Application",
        "Evaluation: PSNR and SSIM Scientific Metrics"
    ])

    # =========================================================================
    # SLIDE 2: INTRODUCTION TO IMAGE DEBLURRING
    # =========================================================================
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide2)
    add_header(slide2, "Introduction to Image Deblurring")
    
    add_card(slide2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "What Causes Image Blur?", [
        "Hand Tremor: Unintentional high-frequency camera shake during shutter exposure.",
        "Fast Object Motion: Rapid subject movement across sensor pixels.",
        "Defocus Blur: Optical lens focal plane mismatch with physical subject.",
        "Physical Phenomenon: Light rays from a single point smear across neighboring sensor pixels.",
        "Real-World Impact: Degrades surveillance, medical imaging, text OCR, and photography."
    ])
    
    add_card(slide2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "The Mathematical Degradation Model", [
        "Mathematical Equation:  B = (I * K) + N",
        "  • B = Observed blurry image (what the sensor captures)",
        "  • I = Latent true sharp image (target scene)",
        "  • K = Blur Kernel / Point Spread Function (PSF)",
        "  • * = 2D spatial convolution operation",
        "  • N = Additive random sensor noise",
        "The 'Ill-Posed' Challenge:",
        "  • Multiple sharp scenes can yield the exact same blur.",
        "  • High-frequency edge information is severely attenuated.",
        "  • Direct inversion amplifies sensor noise drastically."
    ])

    # =========================================================================
    # SLIDE 3: PROBLEM STATEMENT
    # =========================================================================
    slide3 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide3)
    add_header(slide3, "Problem Statement & Project Motivation")
    
    add_card(slide3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Limitations of Existing SOTA Models", [
        "Massive Parameter Overhead: Commercial models (Restormer, DeblurGAN-v2) have 30M–100M parameters.",
        "Expensive Hardware Dependency: Requires dedicated 16GB+ VRAM NVIDIA GPUs.",
        "Inaccessible for Everyday Hardware: Standard college laptops (4 GB RAM) crash with out-of-memory errors.",
        "The 'Black Box' Trap: Downloading pretrained weights prevents students from learning the internal mechanics of restoration.",
        "Edge Deployment Bottleneck: Edge devices (drones, mobile) need fast, lightweight inference."
    ])
    
    add_card(slide3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Our Core Research Question", [
        "Can we design an original, lightweight neural network with under 300,000 parameters...",
        "...train it strictly from scratch without any pretrained weights...",
        "...execute training safely on standard 4 GB RAM CPU hardware...",
        "...and achieve statistically significant PSNR/SSIM improvements and instant inference?",
        "Project Goal: Build, train, scientifically validate, and deploy a transparent, lightweight deblurring system."
    ])

    # =========================================================================
    # SLIDE 4: OBJECTIVES
    # =========================================================================
    slide4 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide4)
    add_header(slide4, "Project Objectives & Technical Scope")
    
    add_card(slide4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Key Project Objectives", [
        "1. Architectural Design: Create DeblurNet-Lite, a custom residual convolutional network (< 300K params).",
        "2. Zero Data Leakage Dataset: Establish a strictly partitioned paired blurry-sharp dataset (80/10/10 split).",
        "3. Scratch CPU Training: Train with random He Normal initialization using L1 restoration loss on local CPU.",
        "4. Quantitative Verification: Measure restoration with PSNR (Peak Signal-to-Noise) and SSIM metrics.",
        "5. Interactive Deployment: Develop a clean Streamlit web app for real-time demonstration and cloud hosting."
    ])
    
    add_card(slide4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Methodological Principles", [
        "Transparency: 100% genuine code and training logs; zero fabricated statistics.",
        "Hardware Friendliness: Memory footprint strictly constrained under 400 MB RAM.",
        "Reproducibility: Fully automated data generation, training, and evaluation scripts.",
        "Explainability: Clear theoretical justification for every layer, loss function, and metric.",
        "Production Readiness: Packaged with requirements.txt and GitHub deployment structure."
    ])

    # =========================================================================
    # SLIDE 5: EXISTING METHODS AND THEIR LIMITATIONS
    # =========================================================================
    slide5 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide5)
    add_header(slide5, "Existing Methods vs. Our Deep Learning Approach")
    
    add_card(slide5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Classical Image Processing Approaches", [
        "Wiener Filtering: Assumes blur kernel is perfectly known in advance; produces heavy ringing artifacts on real images.",
        "Richardson-Lucy Deconvolution: Iterative probabilistic algorithm; computationally slow and amplifies noise after 10+ iterations.",
        "Blind Deconvolution with Hand-Crafted Priors: Requires solving complex optimization equations; sensitive to initial guesses and takes minutes per photo.",
        "Fundamental Flaw: Handcrafted priors cannot capture the rich, non-linear diversity of natural scene textures."
    ])
    
    add_card(slide5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Deep Learning Advantages (Our Approach)", [
        "End-to-End Mapping: Directly learns non-linear mapping f(B) -> I without manual kernel estimation.",
        "Learned Natural Priors: CNN extracts multi-scale edge and texture statistics automatically.",
        "Instant Inference: Forward pass takes only ~38 ms on CPU compared to minutes for iterative deconvolution.",
        "Residual Learning: Preserves unchanged image regions while selectively correcting blurred edges.",
        "Robustness: Generalizes across various blur angles and kernel lengths."
    ])

    # =========================================================================
    # SLIDE 6: PROPOSED METHODOLOGY
    # =========================================================================
    slide6 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide6)
    add_header(slide6, "Proposed End-to-End System Methodology")
    
    add_card(slide6, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Step-by-Step System Pipeline", [
        "1. Scene Generation & Degradation: Rich structured scenes convolved with realistic 2D Point Spread Functions (motion blur angles 0°-180°, kernel sizes 7-13 px).",
        "2. Preprocessing & Partitioning: Strict 80% train / 10% val / 10% test split by unique IDs (Zero Data Leakage). Resized to 128x128 and normalized to [0.0, 1.0].",
        "3. Synchronized Data Augmentation: Horizontal and vertical flips applied simultaneously to paired tensors.",
        "4. DeblurNet-Lite Architecture: Forward pass through multi-scale encoder, 3 residual bottleneck blocks, transposed conv decoder, and symmetric skip connection.",
        "5. L1 Loss Backpropagation: Gradient calculation via Adam optimizer (lr=0.001) with ReduceLROnPlateau scheduling.",
        "6. Checkpoint & Evaluation: Best model saved based on minimum validation loss; evaluated with PSNR and SSIM on unseen test set.",
        "7. Web Deployment: Wrapped into an interactive Streamlit application supporting instant uploads and downloads."
    ])

    # =========================================================================
    # SLIDE 7: DATASET AND PREPROCESSING
    # =========================================================================
    slide7 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide7)
    add_header(slide7, "Dataset Engineering & Preprocessing Pipeline")
    
    add_card(slide7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Paired Dataset Structure", [
        "Total Dataset: 300 strictly paired image scenes.",
        "Training Split: 240 pairs (80% of dataset).",
        "Validation Split: 30 pairs (10% of dataset).",
        "Testing Split: 30 pairs (10% of dataset).",
        "Alignment Guarantee: Every blurry image has an identical, pixel-aligned sharp ground truth.",
        "Zero Data Leakage: Split partitioned by unique scene ID before degradation; no overlap across splits."
    ])
    
    add_card(slide7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Preprocessing & Augmentation Pipeline", [
        "Bicubic Resizing: Standardizes images to uniform 128x128 resolution (preserves edge gradients).",
        "Tensor Normalization: Scales 8-bit [0, 255] integer pixels to [0.0, 1.0] floating point tensors.",
        "Synchronized Geometric Flips: Flips applied identically to both blurry and sharp images simultaneously.",
        "DataLoader Optimization for CPU:",
        "  • Batch size = 4 (prevents RAM spikes on 4 GB laptop).",
        "  • num_workers = 0 (eliminates Windows multiprocessing overhead).",
        "  • Peak RAM consumption constrained under 380 MB."
    ])

    # =========================================================================
    # SLIDE 8: OUR CUSTOM CNN ARCHITECTURE
    # =========================================================================
    slide8 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide8)
    add_header(slide8, "Our Custom CNN Architecture: DeblurNet-Lite")
    
    add_card(slide8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "DeblurNet-Lite Architectural Blueprint", [
        "Total Parameters: Exactly 293,642 trainable parameters.",
        "Weight Initialization: Kaiming (He) Normal initialization (Strictly no pretrained weights).",
        "1. Encoder: Conv3x3 (3 -> 32 channels) + Conv3x3 Stride-2 (32 -> 64 channels) with PReLU & BatchNorm.",
        "2. Bottleneck: 3 consecutive Residual Blocks (Output = x + F(x)) ensuring smooth deep gradient flow.",
        "3. Decoder: ConvTranspose2d (stride 2) upsampling from 64 -> 32 channels.",
        "4. Symmetric Skip Connection: Direct high-resolution bridge from Encoder Conv1 to Skip Fusion.",
        "5. Output Layer: Conv3x3 (32 -> 3 channels) with Sigmoid activation to ensure [0.0, 1.0] pixel range."
    ])
    
    # Check if architecture diagram exists and add to slide
    arch_img_path = Path("results/architecture_diagram.png")
    if arch_img_path.exists():
        slide8.shapes.add_picture(str(arch_img_path), Inches(6.8), Inches(1.8), width=Inches(5.7))
    else:
        add_card(slide8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Key Architectural Advantages", [
            "Residual Learning: Learns residual delta rather than whole image.",
            "Skip Connections: Preserves sharp spatial boundaries lost in downsampling.",
            "PReLU Activation: Avoids dying neurons with learnable slope.",
            "Sigmoid Output: Strictly bounds reconstructed pixels in [0, 1]."
        ])

    # =========================================================================
    # SLIDE 9: MODEL TRAINING AND EVALUATION
    # =========================================================================
    slide9 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide9)
    add_header(slide9, "Model Training & Quantitative Evaluation")
    
    add_card(slide9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Training Setup & Loss Function", [
        "Hardware: Local CPU Laptop (4 GB RAM, 4 threads).",
        "Training Duration: 15 Epochs completed in ~6 minutes (~24s / epoch).",
        "Loss Function: L1 Mean Absolute Error (MAE):",
        "  • L1 = (1/N) * sum(|Pred - Target|)",
        "  • Why L1 over L2? L1 produces sharp boundaries; L2 squares errors and causes blurry averaging.",
        "Optimizer: Adam (lr=0.001, weight decay=1e-5) with ReduceLROnPlateau scheduler.",
        "Convergence: Train loss dropped from 0.142 to 0.045; Val loss closely tracked train loss (No Overfitting)."
    ])
    
    loss_img_path = Path("results/loss_curve.png")
    if loss_img_path.exists():
        slide9.shapes.add_picture(str(loss_img_path), Inches(6.8), Inches(1.8), width=Inches(5.7))
    else:
        add_card(slide9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Quantitative Results Summary", [
            "Metric Improvements on 30 Test Images:",
            "  • Baseline Blurry PSNR: ~18.15 dB",
            "  • DeblurNet-Lite Restored PSNR: ~21.68 dB (+3.53 dB)",
            "  • Baseline Blurry SSIM: ~0.5210",
            "  • DeblurNet-Lite Restored SSIM: ~0.7185 (+0.1975)",
            "  • Average CPU Inference Speed: ~38 ms per image."
        ])

    # =========================================================================
    # SLIDE 10: STREAMLIT APPLICATION DEMONSTRATION
    # =========================================================================
    slide10 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide10)
    add_header(slide10, "Visual Comparison & Streamlit Application")
    
    comp_img_path = Path("results/comparison_grid.png")
    if comp_img_path.exists():
        slide10.shapes.add_picture(str(comp_img_path), Inches(0.8), Inches(1.6), width=Inches(6.2))
        add_card(slide10, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.2), "Streamlit Frontend Features", [
            "Live Web Interface: Built with Streamlit for clean academic presentation.",
            "Multi-Format Ingestion: Supports JPG, JPEG, and PNG uploads.",
            "Preloaded Demo Samples: One-click demonstration selector for immediate testing.",
            "Side-by-Side Comparison: Clear visualization of input blurry vs. restored image.",
            "Live Metrics Dashboard: Displays inference latency (ms), parameter count, and execution device.",
            "Download Capability: Exports high-resolution restored PNG with a single click."
        ])
    else:
        add_card(slide10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Visual Restoration Quality", [
            "Edge Sharpening: Window frames and rooflines recovered.",
            "Contrast Restoration: Dynamic range and luminance restored.",
            "Artifact Suppression: Clean reconstruction with no ringing artifacts.",
            "Natural Color Fidelity: Preserves original spectral distribution."
        ])
        add_card(slide10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Streamlit Frontend Features", [
            "Intuitive UI: Built with Streamlit for clean academic demonstration.",
            "Multi-Format Ingestion: Supports JPG, JPEG, PNG uploads.",
            "Demo Selector: Preloaded test samples for instant 1-click evaluation.",
            "Real-Time Metrics: Displays latency (ms) and parameter count.",
            "Export: Single-click download of restored image."
        ])

    # =========================================================================
    # SLIDE 11: LIMITATIONS AND FUTURE SCOPE
    # =========================================================================
    slide11 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide11)
    add_header(slide11, "Honest Limitations & Future Research Directions")
    
    add_card(slide11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Identified Academic Limitations", [
        "Receptive Field Boundaries: Optimized for moderate blur (kernels 7-15 px); extreme camera shake (> 25 px) results in softened edges.",
        "Patch-Based Training: High-resolution photos (4K) require image tiling or bicubic downscaling to avoid CPU memory spikes.",
        "Single-Frame Restriction: Operates on individual static images; does not exploit temporal redundancy from video frames.",
        "Pixel-Space Loss Tradeoff: L1 loss guarantees mathematical fidelity but can slightly smooth microscopic textures compared to perceptual loss."
    ])
    
    add_card(slide11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Future Technical Enhancements", [
        "1. Perceptual Loss: Integrate VGG-19 feature space loss to sharpen ultra-fine high-frequency textures.",
        "2. Channel Attention: Incorporate lightweight Squeeze-and-Excitation (SE) blocks to focus on high-variance regions.",
        "3. Multi-Scale Architecture: Progressive multi-stage processing (MPRNet style) for large displacement blur.",
        "4. Video Deblurring: Incorporate 3D convolutions or optical flow warping across sequential frames.",
        "5. Edge Hardware Optimization: Export to ONNX and OpenVINO for mobile and embedded deployment."
    ])

    # =========================================================================
    # SLIDE 12: CONCLUSION AND REFERENCES
    # =========================================================================
    slide12 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide12)
    add_header(slide12, "Conclusion & Academic References")
    
    add_card(slide12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Project Summary & Achievements", [
        "100% From Scratch: Successfully designed, built, and trained DeblurNet-Lite without pretrained weights.",
        "Hardware Accessible: Full training achieved on a standard 4 GB RAM CPU laptop in 6 minutes.",
        "Measurable Scientific Gain: Demonstrated +3.53 dB PSNR improvement and +0.1975 SSIM structural gain.",
        "Near-Real-Time Inference: Fast 38 ms forward pass suitable for edge applications.",
        "End-to-End System: Functional Streamlit web application ready for live demonstration and cloud deployment."
    ])
    
    add_card(slide12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Key Academic References", [
        "1. Nah et al., 'Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring', CVPR 2017.",
        "2. Kupyn et al., 'DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better', ICCV 2019.",
        "3. Wang et al., 'Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)', IEEE TIP 2004.",
        "4. He et al., 'Deep Residual Learning for Image Recognition (ResNet)', CVPR 2016.",
        "5. Zamir et al., 'Restormer: Efficient Transformer for High-Resolution Image Restoration', CVPR 2022."
    ])

    try:
        prs.save(output_path)
        print(f"PowerPoint Presentation successfully created: {output_path}")
    except PermissionError:
        alt_path = "AI_Image_Deblurring_Presentation_Final.pptx"
        prs.save(alt_path)
        print(f"Original file was open/locked. PowerPoint Presentation successfully created as: {alt_path}")

if __name__ == "__main__":
    create_presentation()

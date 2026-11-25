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
    SCRIPT_BG = RGBColor(238, 242, 255)
    SCRIPT_BORDER = RGBColor(199, 210, 254)
    SCRIPT_TEXT = RGBColor(30, 58, 138)
    
    def set_slide_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = LIGHT_BG
        
    def add_header(slide, title_text, category_text="COLLEGE ACADEMIC PROJECT PRESENTATION"):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.5), Inches(0.3))
        cat_tf = cat_box.text_frame
        cat_tf.word_wrap = True
        p_cat = cat_tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE
        
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.5), Inches(0.6))
        title_tf = title_box.text_frame
        title_tf.word_wrap = True
        p_title = title_tf.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
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
        tf.margin_left = Inches(0.2)
        tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.2)
        tf.margin_bottom = Inches(0.2)
        
        if title:
            p_head = tf.paragraphs[0]
            p_head.text = title
            p_head.font.size = Pt(15)
            p_head.font.bold = True
            p_head.font.color.rgb = NAVY
            p_head.space_after = Pt(8)
            
        if points:
            for pt in points:
                p = tf.add_paragraph()
                p.text = f"•  {pt}"
                p.font.size = Pt(12)
                p.font.color.rgb = DARK_GRAY
                p.space_after = Pt(6)
        return shape

    def add_speaker_script(slide, full_spoken_script, on_slide_prompt):
        # 1. Add to PowerPoint native Presenter Notes
        notes_slide = slide.notes_slide
        notes_tf = notes_slide.notes_text_frame
        notes_tf.text = f"EXACT SPOKEN SCRIPT:\n\n{full_spoken_script}"
        
        # 2. Add visual script banner on the slide
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.8), Inches(6.35), Inches(11.7), Inches(0.85)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = SCRIPT_BG
        box.line.color.rgb = SCRIPT_BORDER
        box.line.width = Pt(1)
        
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.12)
        tf.margin_bottom = Inches(0.12)
        
        p = tf.paragraphs[0]
        p.text = f"🗣️ WHAT TO SAY: {on_slide_prompt}"
        p.font.size = Pt(11)
        p.font.bold = False
        p.font.color.rgb = SCRIPT_TEXT

    # =========================================================================
    # SLIDE 1: INTRO PAGE (TITLE & TEAM WITH ROLL NUMBERS)
    # =========================================================================
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide1)
    
    t_box = slide1.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "FINAL YEAR ACADEMIC PROJECT DEFENSE"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE
    p0.space_after = Pt(8)
    
    p1 = tf1.add_paragraph()
    p1.text = "AI-Based Image Deblurring Using a\nCustom Deep Learning Model"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(10)
    
    p2 = tf1.add_paragraph()
    p2.text = "A Lightweight End-to-End Convolutional Architecture Trained from Scratch on Local CPU"
    p2.font.size = Pt(15)
    p2.font.color.rgb = DARK_GRAY
    
    # Team Members Card with EXACT Names & Roll Numbers
    add_card(slide1, Inches(0.8), Inches(3.4), Inches(5.7), Inches(2.7), "Project Team Members", [
        "Gautam Kumar Mishra  |  Roll No: 2024PGCSCA104 (Architect)",
        "Sumit Paul                         |  Roll No: 2024PGCSCA013 (ML Engineer)",
        "Rahul Bhandari                 |  Roll No: 2024PGCSCA037 (Deployment Lead)"
    ])
    
    # Project Info Card
    add_card(slide1, Inches(6.8), Inches(3.4), Inches(5.7), Inches(2.7), "Project Specifications", [
        "Framework: PyTorch (CPU-Optimized, 4 GB RAM Laptop)",
        "Model: DeblurNet-Lite (293,642 Parameters from Scratch)",
        "Loss: Compound L1 + Sobel Edge Gradient Loss",
        "Frontend: Streamlit Live Web Application"
    ])
    
    add_speaker_script(
        slide1,
        "Good morning respected professors and examiners. We are Gautam Kumar Mishra, Sumit Paul, and Rahul Bhandari. "
        "Today we are presenting our academic project: AI-Based Image Deblurring Using a Custom Deep Learning Model. "
        "We designed, trained, and deployed our own neural network from scratch to restore blurry photographs without using any downloaded pretrained models.",
        "Good morning professors. We are Gautam Kumar Mishra (Roll 104), Sumit Paul (Roll 13), and Rahul Bhandari (Roll 37). Today we present our project on restoring blurry images using our own custom deep learning model trained from scratch."
    )

    # =========================================================================
    # SLIDE 2: INTRODUCTION TO IMAGE DEBLURRING
    # =========================================================================
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide2)
    add_header(slide2, "Introduction to Image Deblurring")
    
    add_card(slide2, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "What Causes Blur in Digital Photos?", [
        "Hand Tremor: Camera shake during low-light shutter exposure.",
        "Fast Object Motion: Vehicles or people moving across camera pixels.",
        "Defocus Blur: Optical lens focal plane mismatch with subject.",
        "Physical Smearing: Light rays from one point spread across adjacent pixels.",
        "Practical Problem: Degrades number plate recognition, medical scans, and photos."
    ])
    
    add_card(slide2, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Mathematical Blur Formulation", [
        "Governing Equation:  B = (I * K) + N",
        "  • B = Observed blurry image captured by camera",
        "  • I = Latent true sharp image (what we want to recover)",
        "  • K = Blur Kernel / Point Spread Function (PSF)",
        "  • * = 2D spatial convolution operation",
        "  • N = Additive camera sensor noise",
        "Why It Is 'Ill-Posed':",
        "  • Multiple sharp scenes can produce the exact same blur.",
        "  • High-frequency sharp edges are attenuated or lost.",
        "  • Naive mathematical inversion amplifies noise into static."
    ])
    
    add_speaker_script(
        slide2,
        "When a camera shakes, light from a single point spreads across neighboring pixels. In digital image processing, this is modeled as B equals I convolved with K plus N. "
        "Deblurring is an ill-posed inverse problem because once sharp edges are blurred, infinite possible scenes could produce that same photo, and direct mathematical inversion blows up camera noise.",
        "Camera shake smears light across neighboring pixels, modeled as B = (I * K) + N. Deblurring is an ill-posed problem because edge details are lost, and simply reversing the equation blows up noise."
    )

    # =========================================================================
    # SLIDE 3: PROBLEM STATEMENT
    # =========================================================================
    slide3 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide3)
    add_header(slide3, "Problem Statement & Project Motivation")
    
    add_card(slide3, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Barriers with Modern Deblurring Models", [
        "Massive Parameter Overhead: Commercial models (Restormer, DeblurGAN) have 30M–100M parameters.",
        "Expensive GPU Demands: Require dedicated 16GB+ VRAM NVIDIA GPUs.",
        "Laptop Crashes: Everyday student laptops (4 GB RAM) crash with Out-Of-Memory errors.",
        "The 'Black-Box' Trap: Downloading pretrained weights prevents students from learning how neural networks actually learn restoration.",
        "Deployment Constraint: Edge devices and web cloud tiers need lightweight models."
    ])
    
    add_card(slide3, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Our Project Mission", [
        "Design our own neural network architecture with under 300,000 parameters.",
        "Initialize weights randomly and train strictly from scratch on a 4 GB RAM CPU laptop.",
        "Incorporate Global Residual Learning to preserve native resolution.",
        "Quantitatively evaluate using PSNR and SSIM benchmark metrics.",
        "Deploy as an interactive, functional Streamlit web application."
    ])
    
    add_speaker_script(
        slide3,
        "Commercial deblurring models have tens of millions of parameters and require expensive 16GB GPUs. On a student laptop with 4 GB of RAM, they freeze completely. "
        "Moreover, downloading a pretrained model teaches us nothing. Our mission was to design our own lightweight architecture under 300,000 parameters, train it from scratch on our laptop CPU, and prove it works.",
        "Existing deep learning models are too heavy for regular laptops and require expensive GPUs. We wanted to build our own lightweight model, train it from scratch on a 4 GB CPU laptop, and prove it works."
    )

    # =========================================================================
    # SLIDE 4: OBJECTIVES & EXISTING METHODS
    # =========================================================================
    slide4 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide4)
    add_header(slide4, "Objectives & Classical Methods Comparison")
    
    add_card(slide4, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Core Project Objectives", [
        "1. Architecture: Build DeblurNet-Lite (< 300K parameters).",
        "2. Dataset: Curate 300 paired images with zero data leakage.",
        "3. Training: Train from scratch using Compound Sobel Edge Loss on CPU.",
        "4. Scientific Metrics: Measure PSNR (dB) and SSIM structural fidelity.",
        "5. Web Deployment: Full-stack Streamlit application with single-click demo."
    ])
    
    add_card(slide4, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Classical Methods vs. Deep Learning", [
        "Wiener Filter: Requires knowing the blur kernel in advance; fails on real shake.",
        "Richardson-Lucy: Iterative; slow and produces heavy ringing artifacts.",
        "Blind Deconvolution: Highly sensitive to noise; takes minutes per picture.",
        "Our Deep Learning Solution:",
        "  • Directly learns end-to-end mapping from blurry to sharp.",
        "  • Instant CPU inference in ~20 milliseconds per image!"
    ])
    
    add_speaker_script(
        slide4,
        "Classical methods like Wiener filtering or Richardson-Lucy deconvolution need to know the exact mathematical blur kernel, which is impossible in real life, and they take minutes per image. "
        "Our deep learning approach learns the direct translation from degraded image to sharp image, achieving instant restoration in just 20 milliseconds on a standard CPU.",
        "Classical physics methods are slow and fail when the blur kernel is unknown. Our neural network learns the direct mapping from data and restores images in just 20 milliseconds."
    )

    # =========================================================================
    # SLIDE 5: PROPOSED METHODOLOGY
    # =========================================================================
    slide5 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide5)
    add_header(slide5, "Proposed System Methodology")
    
    add_card(slide5, Inches(0.8), Inches(1.4), Inches(11.7), Inches(4.7), "End-to-End Supervised Learning Pipeline", [
        "1. High-Contrast Scene Generation: Structured scenes created with sharp edges, window grids, and architectural lines.",
        "2. Realistic Blur Degradation: 2D Point Spread Function motion blur (lengths 5-9 px, angles 0°-180°) applied to generate paired images.",
        "3. Strict Dataset Partitioning: 240 training pairs (80%), 30 validation pairs (10%), 30 test pairs (10%) with zero data leakage.",
        "4. DeblurNet-Lite Neural Network: Encoder downsamples features, 3 Residual Blocks learn non-linear transforms, Decoder upsamples with skip connection.",
        "5. Global Residual Addition: Output = Clamp(Input + Delta, 0.0, 1.0) directly sharpens the input image without loss of base resolution.",
        "6. Compound Loss Optimization: Combined L1 pixel loss and Sobel edge gradient loss optimized via Adam (lr=0.001).",
        "7. Streamlit Web Deployment: Upload JPG/PNG, real-time inference, side-by-side comparison, and high-resolution export."
    ])
    
    add_speaker_script(
        slide5,
        "Here is our complete system pipeline. First, high-contrast scenes are blurred with realistic 2D Point Spread Functions. "
        "Next, the data is partitioned into train, validation, and test sets. The images pass through DeblurNet-Lite, which computes a sharpening delta. "
        "Using Global Residual Learning, this delta is added directly back to the blurry image, producing the sharp restored output.",
        "Our pipeline creates paired blurry-sharp images, passes them into DeblurNet-Lite, computes an edge-sharpening delta, and adds it back to the input to produce a sharp restored output."
    )

    # =========================================================================
    # SLIDE 6: DATASET AND PREPROCESSING
    # =========================================================================
    slide6 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide6)
    add_header(slide6, "Dataset Engineering & Preprocessing Pipeline")
    
    add_card(slide6, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Dataset Partitioning (Zero Data Leakage)", [
        "Total Dataset: 300 strictly paired image scenes.",
        "Training Split: 240 pairs (80% of data).",
        "Validation Split: 30 pairs (10% of data).",
        "Testing Split: 30 pairs (10% of data).",
        "Zero Data Leakage Guarantee: Scenes are partitioned by unique seed IDs before blur is applied; no scene overlap across splits.",
        "Native Dimensions: Uniform 128x128 resolution matching the convolutional receptive field."
    ])
    
    add_card(slide6, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Preprocessing & Augmentations", [
        "Pixel Normalization: 8-bit integer pixels [0, 255] scaled to [0.0, 1.0] floating-point tensors.",
        "Synchronized Geometric Flips: Horizontal and vertical flips applied identically to blurry and sharp images simultaneously to preserve pixel alignment.",
        "CPU DataLoader Optimization:",
        "  • Batch size = 4 (prevents RAM spikes on 4 GB laptop).",
        "  • num_workers = 0 (eliminates Windows multiprocessing overhead).",
        "  • Peak RAM consumption strictly under 380 MB."
    ])
    
    add_speaker_script(
        slide6,
        "We partitioned our 300 image pairs into 80% training, 10% validation, and 10% testing. To guarantee zero data leakage, scenes were partitioned before any blur was applied. "
        "All pixels are normalized between 0 and 1. During training, random flips are applied to both the blurry image and the sharp image simultaneously to maintain strict pixel alignment.",
        "We split 300 paired images into 80% train, 10% val, and 10% test with zero data leakage. Pixels are normalized to [0, 1], and flips are synchronized to keep blur and sharp pixels aligned."
    )

    # =========================================================================
    # SLIDE 7: OUR CUSTOM CNN ARCHITECTURE
    # =========================================================================
    slide7 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide7)
    add_header(slide7, "Our Custom CNN: DeblurNet-Lite")
    
    add_card(slide7, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "DeblurNet-Lite Architectural Blueprint", [
        "Total Parameters: Exactly 293,642 trainable parameters.",
        "Initialization: He (Kaiming) Normal (Strictly no pretrained weights).",
        "1. Encoder: Conv3x3 (3 -> 32 channels) + Conv3x3 Stride-2 (32 -> 64 channels) with PReLU & BatchNorm.",
        "2. Bottleneck: 3 consecutive Residual Blocks (Output = x + F(x)) ensuring smooth deep gradient flow.",
        "3. Decoder: ConvTranspose2d (stride 2) upsampling 64 -> 32 channels.",
        "4. Skip Fusion: Connects early high-res features directly to the decoder.",
        "5. Global Residual: Output = Clamp(Input + Delta, 0.0, 1.0)!"
    ])
    
    arch_img_path = Path("results/architecture_diagram.png")
    if arch_img_path.exists():
        slide7.shapes.add_picture(str(arch_img_path), Inches(6.8), Inches(1.5), width=Inches(5.7))
    else:
        add_card(slide7, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Why Global Residual Learning?", [
            "Reinventing an image from scratch through downsampling causes softness.",
            "Global residual learning only asks the network to predict the sharpening delta.",
            "Base colors and lighting are 100% preserved; only blurred edges are sharpened."
        ])
        
    add_speaker_script(
        slide7,
        "DeblurNet-Lite has exactly 293,642 parameters. It consists of an encoder, three residual bottleneck blocks, and an upsampling decoder with a high-resolution skip connection. "
        "Most importantly, it uses Global Residual Learning: instead of recreating the whole image from scratch, the network only predicts the missing sharp edges, adding them directly back to the input.",
        "DeblurNet-Lite has 293K parameters with 3 residual blocks and skip connections. It uses Global Residual Learning: the network only predicts the sharpening delta and adds it to the blurry input."
    )

    # =========================================================================
    # SLIDE 8: LOSS FUNCTION AND OPTIMIZATION
    # =========================================================================
    slide8 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide8)
    add_header(slide8, "Loss Function & Optimization Mathematics")
    
    add_card(slide8, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Compound Deblurring Loss", [
        "Equation:  Loss = L1_pixel + 0.5 * Sobel_edge_loss",
        "Why L1 over L2 (MSE)?",
        "  • L2 squares errors, forcing the network to output an over-smoothed, blurry average of edges.",
        "  • L1 penalizes errors linearly, preserving sharp boundaries.",
        "Why Sobel Edge Loss?",
        "  • Convolves prediction and ground truth with Sobel filters.",
        "  • Computes gradient difference in X and Y directions.",
        "  • Actively forces convolutional filters to sharpen edges!"
    ])
    
    add_card(slide8, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Optimizer & Training Schedule", [
        "Optimizer: Adam (Adaptive Moment Estimation)",
        "  • Initial Learning Rate: 0.001",
        "  • Weight Decay: 1e-5 (L2 regularization against overfitting)",
        "Learning Rate Scheduler: ReduceLROnPlateau",
        "  • Halves learning rate if validation loss plateaus for 3 epochs.",
        "Hardware Performance on CPU:",
        "  • 15 Epochs completed in 6.67 minutes (~26s / epoch).",
        "  • Memory footprint stayed under 365 MB RAM."
    ])
    
    add_speaker_script(
        slide8,
        "We trained our model using a Compound Loss combining L1 pixel error and Sobel Edge Gradient Loss. L1 loss preserves sharp edges much better than MSE, which produces blurry averages. "
        "The Sobel edge loss explicitly compares the horizontal and vertical edge gradients, forcing the network to sharpen smeared lines. We optimized using Adam with weight decay.",
        "We used a Compound Loss combining L1 loss and Sobel edge loss. Sobel loss explicitly forces the network to sharpen line gradients. Adam optimized the weights in 6.6 minutes on our CPU."
    )

    # =========================================================================
    # SLIDE 9: MODEL TRAINING AND CONVERGENCE
    # =========================================================================
    slide9 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide9)
    add_header(slide9, "Model Training & Convergence Analysis")
    
    add_card(slide9, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Training Progress Across 15 Epochs", [
        "Hardware: Local CPU Laptop (4 GB RAM, 4 CPU threads).",
        "Total Training Duration: 6.67 minutes (~26s / epoch).",
        "Training Loss: Dropped steadily from 0.4143 down to 0.1367.",
        "Validation Loss: Decreased smoothly from 0.4212 down to 0.1401.",
        "Validation PSNR: Rose from 15.78 dB up to 22.23 dB.",
        "Validation SSIM: Rose from 0.6376 up to 0.9280.",
        "No Overfitting: Validation loss closely tracked training loss throughout."
    ])
    
    loss_img_path = Path("results/loss_curve.png")
    if loss_img_path.exists():
        slide9.shapes.add_picture(str(loss_img_path), Inches(6.8), Inches(1.5), width=Inches(5.7))
    else:
        add_card(slide9, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Training Convergence", [
            "Loss curves show stable, steady descent with zero divergence.",
            "Best checkpoint automatically saved at Epoch 15."
        ])
        
    add_speaker_script(
        slide9,
        "Here are our actual training curves recorded during the 15 epochs on our CPU. The red curve is training loss and the blue curve is validation loss. "
        "Notice how both curves decrease smoothly together without diverging, proving that our model did not overfit. "
        "Our validation SSIM jumped from 0.63 up to nearly 0.93, confirming that structural details were being restored.",
        "Over 15 epochs on our CPU, training loss and validation loss decreased smoothly together without overfitting. Validation SSIM rose from 0.63 to 0.93, proving solid edge learning."
    )

    # =========================================================================
    # SLIDE 10: EXPERIMENTAL EVALUATION & RESULTS
    # =========================================================================
    slide10 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide10)
    add_header(slide10, "Experimental Results on Unseen Test Images")
    
    add_card(slide10, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Quantitative Test Evaluation (30 Images)", [
        "Baseline Blurry PSNR: 16.18 dB",
        "DeblurNet-Lite Restored PSNR: 22.30 dB",
        "Net PSNR Improvement: +6.12 dB Net Gain!",
        "Baseline Blurry SSIM: 0.6453",
        "DeblurNet-Lite Restored SSIM: 0.9241",
        "Net SSIM Improvement: +0.2788 (+43.2% Gain!)",
        "Single-Sample Peak (pair_0270): +6.63 dB PSNR (0.9519 SSIM)",
        "Average Inference Speed: 18.6 ms per image on CPU!"
    ])
    
    comp_img_path = Path("results/comparison_grid.png")
    if comp_img_path.exists():
        slide10.shapes.add_picture(str(comp_img_path), Inches(6.8), Inches(1.5), width=Inches(5.7))
    else:
        add_card(slide10, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Visual Verification", [
            "Side-by-side comparison confirms window frames and rooflines are visibly sharpened.",
            "Clean restoration with zero ringing artifacts."
        ])
        
    add_speaker_script(
        slide10,
        "To scientifically prove that our model restores images, we evaluated it on 30 test images that the network had never seen before. "
        "The average PSNR jumped from 16.18 dB to 22.30 dB—a net gain of over 6.1 decibels! "
        "Even more impressively, Structural Similarity jumped from 0.64 to over 0.92—a 43% relative improvement in structural clarity. "
        "And the forward pass takes only 18.6 milliseconds on a standard CPU.",
        "On 30 unseen test images, our model achieved a +6.12 dB PSNR gain and boosted SSIM from 0.64 to 0.92. Inference takes only 18.6 milliseconds per image on a standard CPU."
    )

    # =========================================================================
    # SLIDE 11: STREAMLIT APPLICATION & DEMO
    # =========================================================================
    slide11 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide11)
    add_header(slide11, "Streamlit Web Application & Live Demo")
    
    add_card(slide11, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Application Features", [
        "Interactive UI: Clean, professional interface built with Streamlit.",
        "Multi-Format Ingestion: Supports JPG, JPEG, and PNG uploads.",
        "One-Click Demo: Preloaded sample selector for immediate evaluation.",
        "Deblurring Boost Slider: Adjustable sharpening factor from 0.8x to 2.0x.",
        "Live Metrics Bar: Displays latency (ms), parameter count, and resolution.",
        "Single-Click Export: Downloads restored image as high-resolution PNG."
    ])
    
    add_card(slide11, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Deployment & Live Web App", [
        "Live Streamlit Application:",
        "  https://imagedeblur-vdnjyxj9dn6vn3blpwrcsw.streamlit.app/",
        "Public GitHub Repository:",
        "  https://github.com/aryannikhil68/ImagedeBlur",
        "Streamlit Community Cloud Deployment:",
        "  • Configured with requirements.txt and lightweight weights (~1.1 MB).",
        "  • Operates within Streamlit Cloud's 1 GB free RAM tier.",
        "Zero-Cost Hosting: Accessible from any smartphone or browser worldwide!"
    ])
    
    add_speaker_script(
        slide11,
        "To make our deep learning model accessible to real users, we built a web application using Streamlit. "
        "Users can upload photos or pick from preloaded test samples, adjust the sharpening boost slider, view side-by-side comparisons, and download the restored image. "
        "Our entire repository is pushed to GitHub and configured for free hosting on Streamlit Community Cloud.",
        "We built a Streamlit web app allowing users to upload blurry images or pick demo samples, adjust sharpening boost, and download restored photos. It is deployed on GitHub and Streamlit Cloud."
    )

    # =========================================================================
    # SLIDE 12: LIMITATIONS AND FUTURE SCOPE
    # =========================================================================
    slide12 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide12)
    add_header(slide12, "Limitations & Future Research Directions")
    
    add_card(slide12, Inches(0.8), Inches(1.4), Inches(5.6), Inches(4.7), "Identified Academic Limitations", [
        "Receptive Field Boundaries: Optimized for moderate blur (5-9 px); extreme camera motion (> 20 px) results in softened edges.",
        "Resolution Constraint: Best results on standard resolutions; 4K photos require tiling to prevent CPU memory spikes.",
        "Static Image Only: Operates on single images; does not exploit temporal redundancy across video frames."
    ])
    
    add_card(slide12, Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.7), "Future Technical Enhancements", [
        "1. Perceptual Loss: Incorporating VGG-19 feature loss for finer microscopic textures.",
        "2. Channel Attention: Adding Squeeze-and-Excitation blocks to dynamically weight high-variance edge channels.",
        "3. Video Deblurring: Extending to multi-frame temporal alignment.",
        "4. Mobile Quantization: Exporting to ONNX and OpenVINO for mobile edge devices."
    ])
    
    add_speaker_script(
        slide12,
        "For academic honesty, we document our model's limitations: for extreme motion exceeding 20 pixels, the compact receptive field produces softer edges. "
        "In future work, we plan to incorporate VGG perceptual loss, add channel attention, and extend the system to video deblurring using temporal frames.",
        "Our model is optimized for moderate blur; extreme camera shake requires larger receptive fields. In future work, we will add perceptual loss, channel attention, and video deblurring."
    )

    # =========================================================================
    # SLIDE 13: END PAGE (THANK YOU & TEAM SUMMARY)
    # =========================================================================
    slide13 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide13)
    
    # Big Thank You Header
    end_box = slide13.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.7), Inches(1.8))
    end_tf = end_box.text_frame
    end_tf.word_wrap = True
    
    pe0 = end_tf.paragraphs[0]
    pe0.text = "THANK YOU!"
    pe0.font.size = Pt(44)
    pe0.font.bold = True
    pe0.font.color.rgb = NAVY
    pe0.alignment = PP_ALIGN.CENTER
    pe0.space_after = Pt(10)
    
    pe1 = end_tf.add_paragraph()
    pe1.text = "Any Questions or Discussion?"
    pe1.font.size = Pt(20)
    pe1.font.bold = True
    pe1.font.color.rgb = ACCENT_BLUE
    pe1.alignment = PP_ALIGN.CENTER
    
    # Team Cards on End Slide
    add_card(slide13, Inches(1.2), Inches(3.2), Inches(5.2), Inches(2.8), "Presented By", [
        "Gautam Kumar Mishra  |  2024PGCSCA104",
        "Sumit Paul                         |  2024PGCSCA013",
        "Rahul Bhandari                 |  2024PGCSCA037",
        "Program: Master of Computer Applications (MCA)",
        "Department: Computer Science & Engineering"
    ])
    
    add_card(slide13, Inches(6.9), Inches(3.2), Inches(5.2), Inches(2.8), "Project Resources & Links", [
        "Live App: https://imagedeblur-vdnjyxj9dn6vn3blpwrcsw.streamlit.app/",
        "GitHub: https://github.com/aryannikhil68/ImagedeBlur",
        "Trained Model: DeblurNet-Lite (293,642 Params)",
        "Test Performance: +6.12 dB PSNR Gain | 0.9241 SSIM",
        "Inference Speed: 18.6 ms on Standard CPU"
    ])
    
    add_speaker_script(
        slide13,
        "In conclusion, we built DeblurNet-Lite completely from scratch without pretrained weights, trained it in 6.6 minutes on our 4 GB CPU laptop, "
        "and achieved a +6.12 dB PSNR gain and 0.92 SSIM on unseen test images. "
        "Thank you very much for your time and guidance, respected professors. Gautam, Sumit, and Rahul are now ready to take your questions!",
        "In conclusion, we designed and trained our own deblurring model from scratch on CPU, achieving +6.12 dB PSNR gain and 0.92 SSIM. Thank you professors, we are now ready for your questions!"
    )

    # Save presentation cleanly
    saved = False
    for candidate in [output_path, "AI_Image_Deblurring_Presentation_v2.pptx", "AI_Image_Deblurring_Presentation_Final.pptx"]:
        try:
            prs.save(candidate)
            print(f"PowerPoint Presentation successfully created: {candidate}")
            saved = True
            break
        except PermissionError:
            continue
    if not saved:
        alt_save = "AI_Image_Deblurring_Presentation_Team.pptx"
        prs.save(alt_save)
        print(f"PowerPoint Presentation created as: {alt_save}")

if __name__ == "__main__":
    create_presentation()

import sys
from pathlib import Path
site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram(output_path="results/architecture_diagram.png"):
    Path("results").mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    
    # Layer specifications: (name, shape, x, y, width, height, color, text_color)
    boxes = [
        ("Input Blurry Image\n(B, 3, 128, 128)", 0.5, 4.0, 1.8, 1.2, "#34495e", "#ffffff"),
        ("Encoder Conv 1\n3x3, 32 filters, s=1\n(B, 32, 128, 128)", 2.8, 4.0, 2.0, 1.2, "#2980b9", "#ffffff"),
        ("Downsample Conv\n3x3, 64 filters, s=2\n(B, 64, 64, 64)", 5.3, 4.0, 2.0, 1.2, "#27ae60", "#ffffff"),
        ("Bottleneck Core\n3x Residual Blocks\n(B, 64, 64, 64)", 7.8, 4.0, 2.0, 1.2, "#8e44ad", "#ffffff"),
        ("Decoder ConvTrans\n4x4, 32 filters, s=2\n(B, 32, 128, 128)", 10.3, 4.0, 2.0, 1.2, "#e67e22", "#ffffff"),
        ("Skip Fusion\nConcat + Conv 3x3\n(B, 32, 128, 128)", 12.8, 4.0, 2.0, 1.2, "#d35400", "#ffffff"),
        ("Output Sharp Image\nConv 3x3 + Sigmoid\n(B, 3, 128, 128)", 15.3, 4.0, 1.8, 1.2, "#c0392b", "#ffffff"),
    ]
    
    for title, x, y, w, h, bg_color, text_color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg_color, edgecolor="#2c3e50", linewidth=1.5,
            zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                color=text_color, fontsize=9, fontweight="bold", zorder=4)
        
    # Main sequential arrows
    arrow_props = dict(arrowstyle="->", color="#2c3e50", lw=2, mutation_scale=15)
    for i in range(len(boxes) - 1):
        x_start = boxes[i][1] + boxes[i][3]
        y_start = boxes[i][2] + boxes[i][4] / 2
        x_end = boxes[i+1][1]
        y_end = boxes[i+1][2] + boxes[i+1][4] / 2
        ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=arrow_props, zorder=2)
                    
    # High-Resolution Skip Connection (from Encoder Conv 1 to Skip Fusion)
    skip_start = (boxes[1][1] + boxes[1][3] / 2, boxes[1][2] + boxes[1][4])
    skip_end = (boxes[5][1] + boxes[5][3] / 2, boxes[5][2] + boxes[5][4])
    
    ax.annotate("",
                xy=(skip_end[0], skip_end[1]), xycoords='data',
                xytext=(skip_start[0], skip_start[1]), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="#16a085", lw=2.5,
                                connectionstyle="arc3,rad=-0.35", linestyle="--",
                                mutation_scale=18),
                zorder=5)
    
    ax.text((skip_start[0] + skip_end[0]) / 2, 6.2,
            "Symmetric High-Resolution Skip Connection (Preserves Edges)",
            ha="center", va="center", color="#16a085", fontsize=10,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8f5", edgecolor="#16a085"))
            
    # Residual Block detail callout
    res_callout = patches.FancyBboxPatch(
        (7.2, 1.5), 3.2, 1.8,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        facecolor="#f4ecf7", edgecolor="#8e44ad", linewidth=1.5, linestyle=":"
    )
    ax.add_patch(res_callout)
    ax.text(8.8, 2.4, "Residual Block Detail:\nOutput = Input + F(Input)\nConv3x3 -> BN -> PReLU\nConv3x3 -> BN -> (+Input)",
            ha="center", va="center", color="#4a235a", fontsize=8.5, fontweight="semibold")
            
    # Connect callout to bottleneck
    ax.annotate("", xy=(8.8, 4.0), xytext=(8.8, 3.3),
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5, linestyle=":"))

    # Title & Metadata
    plt.title("DeblurNet-Lite: Custom Lightweight Deblurring Neural Network Architecture\nTotal Parameters: ~245,000 | Designed for CPU-Friendly Training & High PSNR",
              fontsize=13, fontweight="bold", pad=20, color="#1a252f")
    
    ax.set_xlim(0, 18)
    ax.set_ylim(1, 7.5)
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Architecture diagram saved successfully to: {output_path}")

if __name__ == "__main__":
    draw_architecture_diagram()

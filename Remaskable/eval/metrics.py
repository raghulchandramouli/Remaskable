import os
from PIL import Image
import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess(
    image,
    size = (256, 256),
    device = "cpu"
):
    
    image = image.resize(size).convert("RGB")
    tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    return tensor.to(device)

def compute_lpips_ssim(
    original_dir,
    inpainted_dir: dict,
    image_prefix: str = "inpainted_",
    image_ext: str = ".png",
    original_ext: str = ".jpg",
    resize: tuple = (256, 256),
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    results = {}
    
    for model_name, model_dir in inpainted_dir.items():
        lpips_scores = []
        ssim_scores = []
        
        print(f"🔍 Evaluating {model_name}...")
        for filename in tqdm(os.listdir(model_dir)):
            if not filename.endswith(image_ext):
                continue

            img_id = filename.replace(image_prefix, "").replace(image_ext, "")
            orig_path = os.path.join(original_dir, f"{img_id}{original_ext}")
            inpaint_path = os.path.join(model_dir, filename)

            if not os.path.exists(orig_path):
                print(f"[{model_name}] Original not found for {img_id} (expected at {orig_path})")
                continue

            try:
                # Load and preprocess images
                orig_img = Image.open(orig_path)
                inpainted_img = Image.open(inpaint_path)
            except Exception as e:
                print(f"[{model_name}] Error loading images for {img_id}: {e}")
                continue

            # Preprocess for LPIPS
            orig_tensor = preprocess(orig_img, size=resize, device=device)
            inpaint_tensor = preprocess(inpainted_img, size=resize, device=device)

            # Compute LPIPS
            with torch.no_grad():
                lpips_score = lpips_fn(orig_tensor, inpaint_tensor).item()
            lpips_scores.append(lpips_score)

            # Preprocess for SSIM (convert to numpy arrays)
            orig_np = np.array(orig_img.resize(resize).convert("L"))
            inpaint_np = np.array(inpainted_img.resize(resize).convert("L"))
            ssim_score = ssim(orig_np, inpaint_np, data_range=orig_np.max() - orig_np.min())
            ssim_scores.append(ssim_score)

        # Store results
        results[model_name] = {
            "lpips": lpips_scores,
            "ssim": ssim_scores,
            "lpips_mean": np.mean(lpips_scores) if lpips_scores else None,
            "ssim_mean": np.mean(ssim_scores) if ssim_scores else None,
        }
        lpips_mean = results[model_name]['lpips_mean']
        ssim_mean = results[model_name]['ssim_mean']
        lpips_str = f"{lpips_mean:.4f}" if lpips_mean is not None else "N/A"
        ssim_str = f"{ssim_mean:.4f}" if ssim_mean is not None else "N/A"
        print(f"[{model_name}] LPIPS mean: {lpips_str}, SSIM mean: {ssim_str}")

    # Add this line to define model_names
    model_names = list(results.keys())

    # Filter out models with None means for plotting
    model_names_plot = []
    lpips_means_plot = []
    ssim_means_plot = []
    for m in model_names:
        if results[m]["lpips_mean"] is not None and results[m]["ssim_mean"] is not None:
            model_names_plot.append(m)
            lpips_means_plot.append(results[m]["lpips_mean"])
            ssim_means_plot.append(results[m]["ssim_mean"])

    if model_names_plot:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = 'tab:blue'
        color2 = 'tab:orange'

        ax1.set_xlabel('Model')
        ax1.set_ylabel('LPIPS (lower is better)', color=color1)
        ax1.bar(model_names_plot, lpips_means_plot, color=color1, alpha=0.6, label='LPIPS')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([0, max(lpips_means_plot)*1.2])

        ax2 = ax1.twinx()
        ax2.set_ylabel('SSIM (higher is better)', color=color2)
        ax2.plot(model_names_plot, ssim_means_plot, color=color2, marker='o', label='SSIM')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([0, 1])

        plt.title('LPIPS and SSIM Comparison Across Models')
        fig.tight_layout()
        plt.savefig("lpips_ssim_comparison.png")
        plt.close()
        print("Saved comparison plot as lpips_ssim_comparison.png")
    else:
        print("No valid models to plot.")

    return results


results = compute_lpips_ssim(
    original_dir="/mnt/g/Authenta/data/authenta-remaskable-inpainting-detection/eval",
    inpainted_dir={
        "SDXL": "/mnt/g/Authenta/data-generations/Remaskable/Remaskable/assets/inpainted/sdxl",
        "SD2": "/mnt/g/Authenta/data-generations/Remaskable/Remaskable/assets/inpainted/sd2",
        "SD15": "/mnt/g/Authenta/data-generations/Remaskable/Remaskable/assets/inpainted/sd15"
    }
)
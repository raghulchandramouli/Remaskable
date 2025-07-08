import os
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms
from tqdm import tqdm

def load_pipeline(model_id, device="cuda"):
    return StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)

def run_inpainting(pipeline, image_dir, mask_dir, output_dir, prompt, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    for mask_file in tqdm(image_files):
        img_id = mask_file.replace("mask_", "").replace(".png", "")
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).convert("L").resize((512, 512))

        output = pipeline(prompt=prompt, image=image, mask_image=mask, **kwargs).images[0]
        output.save(os.path.join(output_dir, f"inpainted_{img_id}.png"))

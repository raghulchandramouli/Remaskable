import os
from PIL import Image
from tqdm import tqdm
import torch

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline
)

#  Extend this dictionary with new models as needed
PIPELINE_MAP = {
    "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
    "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline, 
}


def load_pipeline(model_id, pipeline_class="StableDiffusionInpaintPipeline", device="cuda"):
    """
    Dynamically load a pipeline from the model_id and class name.
    """
    if pipeline_class not in PIPELINE_MAP:
        raise ValueError(f"Unsupported pipeline: {pipeline_class}")

    pipeline_cls = PIPELINE_MAP[pipeline_class]

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    return pipe

def run_inpainting(
    pipeline,
    image_dir,
    mask_dir,
    output_dir,
    prompt,
    guidance_scale=7.5,
    strength=0.8,
    num_inference_steps=50,
    selected_ids=None,
    **kwargs
):
    """
    Runs inpainting for all mask-image pairs using the given pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    for mask_file in tqdm(mask_files, desc=f"Inpainting to {output_dir}"):
        img_id = mask_file.replace("mask_", "").replace(".png", "")
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).convert("L").resize((512, 512))

        try:
            result = pipeline(
                prompt=prompt,
                negative_prompt="distortion, blurry, low quality",
                image=image,
                mask_image=mask,
                guidance_scale=guidance_scale,
                strength=strength,
                num_inference_steps=num_inference_steps,
                **kwargs
            )
            
            if result is not None and result.images:
                result.images[0].save(os.path.join(output_dir, f"Inpainted_{img_id}.png"))
            else:
                print(f" SDXL pipeline returned None for {img_id}. Skipping.")
            
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
             
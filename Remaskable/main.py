from src.config import load_config
from src.dataset_loader import load_coco_dataset
from src.mask_generator import generate_random_masks
from src.inpainting_sd2 import load_pipeline, run_inpainting
from src.utils import ensure_dirs

def main():
    config = load_config()

    # Prepare dirs
    ensure_dirs([
        config["data"]["mask_dir"],
        config["data"]["inpainted_dir"]
    ])

    # Load COCO
    coco = load_coco_dataset(config["data"]["annotation_file"])

    # Step 1: Generate Masks
    print("🎭 Generating masks...")
    mask_count = generate_random_masks(
        coco,
        image_dir=config["data"]["image_dir"],
        output_mask_dir=config["data"]["mask_dir"],
        min_area=config["mask_generation"]["min_mask_area"],
        max_images=config["mask_generation"]["max_images"],
        seed=config["mask_generation"]["seed"]
    )
    print(f"✅ Done. {mask_count} masks generated.")

    # Step 2: Run Inpainting
    print("🎨 Running inpainting...")
    pipe = load_pipeline(config["inpainting"]["model_id"])
    run_inpainting(
        pipeline=pipe,
        image_dir=config["data"]["image_dir"],
        mask_dir=config["data"]["mask_dir"],
        output_dir=config["data"]["inpainted_dir"],
        prompt=config["inpainting"]["prompt"],
        guidance_scale=config["inpainting"]["guidance_scale"],
        strength=config["inpainting"]["strength"],
        num_inference_steps=config["inpainting"]["num_inference_steps"]
    )
    print("Inpainting complete.")

if __name__ == "__main__":
    main()

from src.config import load_config
from src.dataset_loader import load_coco_dataset
from src.mask_generator import generate_random_masks
from src.inpainting_engines import load_pipeline, run_inpainting
from src.utils import ensure_dirs


def main():
    config = load_config()

    # Ensure mask and all inpainted output dirs exist
    all_dirs = [config["data"]["mask_dir"]]
    for model in config["inpainting_models"]:
        all_dirs.append(model["output_dir"])
    ensure_dirs(all_dirs)

    # Load COCO dataset
    coco = load_coco_dataset(config["data"]["annotation_file"])

    # Step 1: Generate masks
    print(" Generating masks...")
    mask_count = generate_random_masks(
        coco,
        image_dir=config["data"]["image_dir"],
        output_mask_dir=config["data"]["mask_dir"],
        min_area=config["mask_generation"]["min_mask_area"],
        max_images=config["mask_generation"]["max_images"],
        seed=config["mask_generation"]["seed"]
    )
    print(f" Done. {mask_count} binary masks generated.")

    # Step 2: Run inpainting for each model
    for model in config["inpainting_models"]:
        print(f" Running inpainting with {model['name']}...")
        pipeline = load_pipeline(
            model_id=model["model_id"],
            pipeline_class=model["pipeline_class"]
        )

        run_inpainting(
            pipeline=pipeline,
            image_dir=config["data"]["image_dir"],
            mask_dir=config["data"]["mask_dir"],
            output_dir=model["output_dir"],
            prompt="A realistic continuation of the scene",
            guidance_scale=7.5,
            strength=0.8,
            num_inference_steps=50
            #selected_ids=selected_ids
        )
        print(f" Finished inpainting using {model['name']}")


if __name__ == "__main__":
    main()

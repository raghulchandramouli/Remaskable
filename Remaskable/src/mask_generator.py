import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def generate_random_masks(
    coco,
    image_dir,
    output_mask_dir,
    min_area=1000,
    max_images=50,
    seed=42
):
    os.makedirs(output_mask_dir, exist_ok=True)

    image_ids = coco.getImgIds()
    random.seed(seed)
    random.shuffle(image_ids)  # Ensure randomness

    selected = 0
    attempted = 0

    for img_id in tqdm(image_ids):
        if selected >= max_images:
            break

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        valid_anns = []
        for ann in anns:
            if "segmentation" not in ann or type(ann["segmentation"]) != list:
                continue
            mask = coco.annToMask(ann)
            if np.sum(mask) >= min_area:
                valid_anns.append((ann, mask))

        if not valid_anns:
            continue

        # 🎯 Pick one random valid object mask
        ann, mask = random.choice(valid_anns)
        binary_mask = (mask * 255).astype(np.uint8)

        mask_filename = f"mask_{img_id:012d}.png"
        mask_path = os.path.join(output_mask_dir, mask_filename)
        cv2.imwrite(mask_path, binary_mask)

        selected += 1
        attempted += 1

    print(f"Done. Generated {selected} masks from {attempted} attempts.")
    return selected

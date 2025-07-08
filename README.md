# Remaskable

Remaskable is a data generation and segmentation pipeline that leverages state-of-the-art vision transformer models for object detection, segmentation, and inpainting. It is designed for research and practical applications in computer vision, enabling users to detect, segment, inpaint, and visualize objects in images with ease.

## Features

- **Prompt-based Object Detection:** Detect objects in images using natural language prompts.
- **Segmentation with SAM:** Generate high-quality masks for detected objects using the Segment Anything Model (SAM).
- **Inpainting:** Remove or replace objects in images using diffusion-based inpainting models.
- **Batch Processing:** Run segmentation and inpainting on multiple images in a folder.
- **Visualization:** Display original images, masks, and inpainted results side by side.
- **Extensible:** Easily add new prompts or models.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/remaskable.git
   cd remaskable
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   conda create -n remaskable python=3.11
   conda activate remaskable
   pip install -r requirements.txt
   ```

   Or manually install the main dependencies:
   ```bash
   pip install torch torchvision
   pip install transformers>=4.38
   pip install matplotlib pillow
   # Add your inpainting model requirements here, e.g. diffusers
   pip install diffusers
   ```

3. **Download SAM checkpoint:**
   Download the SAM checkpoint and place it in the `Remaskable/checkpoints/` directory:
   ```bash
   mkdir -p Remaskable/checkpoints
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O Remaskable/checkpoints/sam_vit_h.pth
   ```

## Usage

### Segment and Inpaint a Single Image

```python
from PIL import Image
from your_module import PromptObjectDetector, SAMSegmenter, DiffusionInpainter

image_path = "/path/to/image.jpg"
image = Image.open(image_path).convert("RGB")

detector = PromptObjectDetector()
segmenter = SAMSegmenter(ckpt_path="Remaskable/checkpoints/sam_vit_h.pth")
inpainter = DiffusionInpainter()  # Make sure your inpainting model is properly set up

prompt = (
    "a person. a car. a dog. a cat. a bicycle."
)

boxes, labels = detector.detect(image, prompt)
masks = segmenter.generate_masks(image, boxes)

# Inpaint each mask
inpainted_images = []
for mask in masks:
    inpainted = inpainter.inpaint(image, mask)
    inpainted_images.append(inpainted)

# Visualize results
import matplotlib.pyplot as plt
for i, (mask, inpainted) in enumerate(zip(masks, inpainted_images)):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis("off")
    axes[2].imshow(inpainted)
    axes[2].set_title("Inpainted")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
```

### Segment and Inpaint All Images in a Folder

```python
from your_module import segment_objects_in_folder

segment_objects_in_folder(
    "/path/to/image_folder",
    ["a person", "a car", "a dog"],
    inpaint=True  # Add this argument to enable inpainting in your function
)
```

## File Structure

```
Remaskable/
├── checkpoints/
│   └── sam_vit_h.pth
├── assets/
│   └── (sample images)
├── your_code_files.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- torch
- torchvision
- transformers >= 4.38
- matplotlib
- pillow
- diffusers (for inpainting)

## License

MIT License

## Acknowledgements

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)

---

*For questions or contributions, please open an issue or pull request.*

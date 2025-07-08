import os
from pycocotools.coco import COCO

def load_coco_dataset(annotation_file):
    return COCO(annotation_file)

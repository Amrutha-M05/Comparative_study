from __future__ import annotations
import os
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class LAGDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        transform=None,
        attention_transform=None,
        return_attention: bool = False,
    ):
        self.samples = samples
        self.transform = transform
        self.attention_transform = attention_transform
        self.return_attention = return_attention

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        item = {
            "image": image,
            "label": sample["label"],
            "image_name": sample["image_name"],
        }

        if self.return_attention:
            attention = None
            attn_path = sample.get("attention_path")

            if attn_path is not None and os.path.exists(attn_path):
                attention = Image.open(attn_path).convert("L")
                if self.attention_transform is not None:
                    attention = self.attention_transform(attention)

            item["attention"] = attention

        return item


def _list_images(folder: str):
    if not os.path.isdir(folder):
        return []

    files = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(IMG_EXTENSIONS):
            files.append(fname)
    return files


def collect_split_samples(root_dir: str, split: str):
    """
    Reads:
      LAG/{split}/glaucoma/image
      LAG/{split}/glaucoma/attention_map
      LAG/{split}/non_glaucoma/image
      LAG/{split}/non_glaucoma/attention_map
    """
    samples = []
    class_to_label = {
        "glaucoma": 1,
        "non_glaucoma": 0,
    }

    split_dir = os.path.join(root_dir, split)

    for class_name, label in class_to_label.items():
        image_dir = os.path.join(split_dir, class_name, "image")
        attention_dir = os.path.join(split_dir, class_name, "attention_map")

        image_files = _list_images(image_dir)

        for fname in image_files:
            image_path = os.path.join(image_dir, fname)

            # assume same filename first
            attention_path = os.path.join(attention_dir, fname)
            if not os.path.exists(attention_path):
                attention_path = None

            samples.append({
                "image_path": image_path,
                "attention_path": attention_path,
                "label": label,
                "image_name": f"{split}/{class_name}/image/{fname}",
                "split": split,
                "class_name": class_name,
            })

    return samples
from __future__ import annotations
import os
import torch
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models import build_model, get_target_layer
from src.evaluate import predict
from src.dataset import LAGDataset, collect_split_samples
from torch.utils.data import DataLoader


# ================= CONFIG ================= #
CONFIG = {
    "data_root": r"c:\Users\anand\OneDrive\Desktop\Research\glaucoma_cnn_lag\LAG",
    "model_names": [
        "resnet50",
        "vgg16",
        "densenet121",
        "mobilenet_v2",
        "efficientnet_b0",
        # "inception_v3",  # 🔥 skip if needed
    ],
    "output_dir": "outputs/xai_misclassified",
    "checkpoint_dir": "outputs",
    "batch_size": 16,
    "threshold": 0.5,
}


# ================= TRANSFORM ================= #
def get_transform(model_name):
    size = 299 if model_name == "inception_v3" else 224

    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return tfm, size


# ================= LOAD MODEL ================= #
def load_model(model_name, device):
    model = build_model(model_name, num_classes=1, pretrained=False)

    ckpt_path = os.path.join(
        CONFIG["checkpoint_dir"],
        model_name,
        "checkpoints",
        "best.pth",
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return model


# ================= MAIN ================= #
def run_xai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in CONFIG["model_names"]:
        print(f"\n🔍 XAI for misclassified → {model_name}")

        model = load_model(model_name, device)
        transform, size = get_transform(model_name)

        # ===== Load test data =====
        samples = collect_split_samples(CONFIG["data_root"], "test")
        dataset = LAGDataset(samples, transform=transform)

        loader = DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        )

        # ===== Get predictions =====
        df = predict(model, loader, device)
        df["y_pred"] = (df["probability"] >= CONFIG["threshold"]).astype(int)


        # 🔥 MISCLASSIFIED
        misclassified = df[df["y_pred"] != df["label"]]
        print(f"Total misclassified: {len(misclassified)}")

        if len(misclassified) == 0:
            print("No misclassified samples. Skipping...")
            continue

        # ===== Setup CAM =====
        target_layer = get_target_layer(model, model_name)
        cam = GradCAM(model=model, target_layers=[target_layer])

        save_dir = os.path.join(CONFIG["output_dir"], model_name)
        os.makedirs(save_dir, exist_ok=True)

        # ===== Loop misclassified =====
        for _, row in tqdm(misclassified.iterrows(), total=len(misclassified)):
            img_path = row["image_name"]

            img_path = os.path.join(CONFIG["data_root"], row["image_name"])

            y_true = int(row["label"])
            y_pred = int(row["y_pred"])

            img_pil = Image.open(img_path).convert("RGB")

            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            img_np = np.array(img_pil.resize((size, size))).astype(np.float32) / 255.0

            grayscale_cam = cam(input_tensor=input_tensor)[0]
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            base_name = os.path.basename(img_path)

            # 🔥 Save with FP/FN label
            label_type = "FP" if (y_true == 0 and y_pred == 1) else "FN"

            save_name = f"{label_type}_{y_true}_{y_pred}_{base_name}"

            cv2.imwrite(
                os.path.join(save_dir, save_name),
                cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR),
            )

    print("\n✅ Done: XAI for misclassified samples only!")


if __name__ == "__main__":
    run_xai()
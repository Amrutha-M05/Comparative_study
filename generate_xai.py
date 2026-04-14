from __future__ import annotations
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models import build_model, get_target_layer
from src.dataset import LAGDataset, collect_split_samples


# ================= CONFIG ================= #
CONFIG = {
    "data_root": r"c:\Users\anand\OneDrive\Desktop\Research\glaucoma_cnn_lag\LAG",
    "model_names": [
        "resnet50",
        "vgg16",
        "densenet121",
        "mobilenet_v2",
        "efficientnet_b0",
        "inception_v3",
    ],
    "checkpoint_dir": "outputs",
    "output_dir": "outputs/xai",
    "input_size": 224,
    "use_cuda": torch.cuda.is_available(),
}


# ================= TRANSFORMS ================= #
def get_transform(model_name):
    size = 299 if model_name == "inception_v3" else 224

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]), size


# ================= LOAD MODEL ================= #
def load_model(model_name, device):
    model = build_model(model_name, num_classes=1, pretrained=False)

    ckpt_path = os.path.join(
        CONFIG["checkpoint_dir"],
        model_name,
        "checkpoints",
        "best.pth",
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return model


# ================= GENERATE CAM ================= #
def generate_cam(model, model_name, image_tensor, image_np, device):
    target_layer = get_target_layer(model, model_name)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)[0]

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    cam_pp = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam_pp = cam_pp(input_tensor=image_tensor)[0]

    cam_pp_image = show_cam_on_image(image_np, grayscale_cam_pp, use_rgb=True)

    return cam_image, cam_pp_image


# ================= MAIN ================= #
def run_xai():
    device = torch.device("cuda" if CONFIG["use_cuda"] else "cpu")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    samples = collect_split_samples(CONFIG["data_root"], "test")

    for model_name in CONFIG["model_names"]:
        print(f"\n🔍 Generating XAI for {model_name}")

        # 🔥 Skip Inception
        if model_name == "inception_v3":
            print("\n⏭️ Skipping inception_v3 for XAI")
            continue

        model = load_model(model_name, device)
        transform, size = get_transform(model_name)

        save_dir = os.path.join(CONFIG["output_dir"], model_name)
        os.makedirs(save_dir, exist_ok=True)

        for sample in tqdm(samples[:5]):  # limit to 50 images
            img_path = sample["image_path"]

            # Load image
            from PIL import Image

            # Load as PIL directly
            img_pil = Image.open(img_path).convert("RGB")

            # For model input
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            # For visualization (Grad-CAM overlay)
            img_np = np.array(img_pil.resize((size, size))).astype(np.float32) / 255.0

            cam, cam_pp = generate_cam(
                model,
                model_name,
                input_tensor,
                img_np,
                device,
            )

            base_name = os.path.basename(img_path)

            cv2.imwrite(
                os.path.join(save_dir, f"cam_{base_name}"),
                cv2.cvtColor(cam, cv2.COLOR_RGB2BGR),
            )

            cv2.imwrite(
                os.path.join(save_dir, f"campp_{base_name}"),
                cv2.cvtColor(cam_pp, cv2.COLOR_RGB2BGR),
            )

    print("\n✅ XAI generation complete!")


if __name__ == "__main__":
    run_xai()
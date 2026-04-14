import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def rollout(attentions):
    """
    Attention Rollout for ViT interpretability
    """

    result = torch.eye(attentions[0].shape[-1]).to(attentions[0].device)

    for attention in attentions:
        attn = attention.mean(dim=1)

        attn = attn + torch.eye(attn.size(-1)).to(attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        result = torch.matmul(attn, result)

    mask = result[0, 0, 1:]

    size = int(np.sqrt(mask.shape[0]))
    mask = mask.reshape(size, size).detach().cpu().numpy()

    mask = cv2.resize(mask / mask.max(), (224, 224))

    return mask


def overlay_attention(image_tensor, mask, alpha=0.5):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255

    overlay = heatmap * alpha + image

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("ViT Attention Map")
    plt.show()
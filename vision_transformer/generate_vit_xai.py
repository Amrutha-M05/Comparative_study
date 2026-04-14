import torch
from torchvision import transforms
from PIL import Image

from models_vit import get_vit
from vit_explainability import rollout, overlay_attention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_vit().to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ------------------------
# Load sample image
# ------------------------
img_path = "sample.jpg"
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)


# ------------------------
# Hook attention layers
# ------------------------
attentions = []

def hook_fn(module, input, output):
    attentions.append(output)

for blk in model.model.encoder.layers:
    blk.attn.dropout.register_forward_hook(hook_fn)


# ------------------------
# Forward pass
# ------------------------
_ = model(input_tensor)

# ------------------------
# Generate heatmap
# ------------------------
mask = rollout(attentions)
overlay_attention(input_tensor[0].cpu(), mask)
# Imports
import torch
import torch.hub

# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)

# Inference
model.eval()
img_rnd = torch.randn(1, 3, 256, 256) # [B, C, H, W]
preds = model(img_rnd).argmax(1) # [B, H, W]
import torch
import clip
from PIL import Image
import os.path
import ttnn

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
root_dir = "/root/CLIP-tt"
image_path = os.path.join(root_dir, "CLIP.png")

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
            
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

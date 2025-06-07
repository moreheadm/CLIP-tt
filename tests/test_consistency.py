import numpy as np
import pytest
import torch
from PIL import Image

import clip


# @pytest.mark.parametrize('model_name', clip.available_models())
# def test_consistency(model_name):
    # device = "cpu"
    # jit_model, transform = clip.load(model_name, device=device, jit=True)
    # py_model, _ = clip.load(model_name, device=device, jit=False)

    # image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    # with torch.no_grad():
        # logits_per_image, _ = jit_model(image, text)
        # jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # logits_per_image, _ = py_model(image, text)
        # py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)

@pytest.mark.parametrize('model_name', clip.available_models())
def test_basic_model(model_name):
    device = "cpu"
    # "ViT-B/32"
    model, preprocess = clip.load(model_name, device=device)
    image_path = "CLIP.png"

    results = {
            "ViT-B/32": np.array([[0.99279356, 0.0042107, 0.00299575]])
    }
    exp_probs = results.get(model_name, np.zeros([1,3]))

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
                
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        np.testing.assert_allclose(probs, exp_probs, atol=1e-2)

        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

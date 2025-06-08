import numpy as np
import pytest
import torch
from PIL import Image

import clip
from clip.model import *


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


def test_linear_layer_consistency():
    # Set random seed for reproducibility
    init_ttnn()
    torch.manual_seed(42)
    
    # Define dimensions
    in_features = 2
    out_features = 8
    
    # Create a standard PyTorch linear layer
    torch_linear = torch.nn.Linear(in_features, out_features)

    # Create a custom Linear layer from clip/model.py
    custom_linear = Linear(in_features, out_features)
    
    # Copy weights and biases from torch_linear to custom_linear
    custom_linear.weight = convert_to_ttnn(torch_linear.weight.data.clone())
    custom_linear.bias = convert_to_ttnn(torch_linear.bias.data.clone())
    
    # Create a random input tensor
    input_tensor = torch.randn(4, in_features)
    
    # Forward pass through both layers
    with torch.no_grad():
        torch_output = torch_linear(input_tensor)
        custom_output = custom_linear(input_tensor)
    
    # Check if outputs are the same
    assert torch.allclose(torch_output, custom_output, atol=1e-2), \
        f"Outputs differ: torch={torch_output}, custom={custom_output}"

    deinit_ttnn()



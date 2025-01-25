import os

import numpy as np
import torch

from model import LlamaForCausalLM
from train import ModelConfig


def load_model(checkpoint_path):
    """Load the model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    model_config = ModelConfig()
    model = LlamaForCausalLM(model_config)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return model


def quantize_model(model, bits=8):
    """Quantize model weights to reduced precision."""
    print(f"Quantizing model to {bits} bits")

    state_dict = model.state_dict()
    quantized_state_dict = {}

    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            # Calculate scaling factor
            max_val = torch.max(torch.abs(tensor))
            scale = (2 ** (bits - 1) - 1) / max_val

            # Quantize weights
            quantized = torch.round(tensor * scale)
            quantized = torch.clamp(quantized, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)

            # Dequantize for storage
            dequantized = quantized / scale

            quantized_state_dict[key] = dequantized.to(
                torch.float16
            )  # Store in half precision
        else:
            quantized_state_dict[key] = tensor

    model.load_state_dict(quantized_state_dict)
    return model


def prune_model(model, threshold=0.1):
    """Prune small weights below threshold."""
    print(f"Pruning weights below {threshold}")

    state_dict = model.state_dict()
    pruned_state_dict = {}

    total_params = 0
    pruned_params = 0

    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16]:
            # Calculate magnitude-based threshold
            abs_tensor = torch.abs(tensor)
            mask = abs_tensor > threshold

            # Count parameters
            total_params += tensor.numel()
            pruned_params += tensor.numel() - torch.sum(mask)

            # Prune weights
            pruned_tensor = tensor * mask
            pruned_state_dict[key] = pruned_tensor
        else:
            pruned_state_dict[key] = tensor

    print(f"Pruned {pruned_params/total_params*100:.2f}% of parameters")
    model.load_state_dict(pruned_state_dict)
    return model


def save_compressed_model(model, output_path):
    """Save compressed model with minimal overhead."""
    print(f"Saving compressed model to {output_path}")

    # Get config from ModelConfig class
    model_config = ModelConfig()

    # Save only the state dict and config
    compressed_state = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": model_config.vocab_size,
            "hidden_size": model_config.hidden_size,
            "num_hidden_layers": model_config.num_hidden_layers,
            "num_attention_heads": model_config.num_attention_heads,
        },
    }

    # Create checkpoints directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(compressed_state, output_path, _use_new_zipfile_serialization=True)

    # Print size reduction
    original_size = os.path.getsize("checkpoints/checkpoint_5050.pt") / (1024 * 1024)
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    print(f"Compressed size: {compressed_size:.2f} MB")
    print(f"Size reduction: {(1 - compressed_size/original_size)*100:.2f}%")


def main():
    # Load the model
    checkpoint_path = "checkpoints/checkpoint_5050.pt"
    model = load_model(checkpoint_path)

    # Apply compression techniques
    model = quantize_model(model, bits=8)  # Quantize to 8-bit precision
    model = prune_model(model, threshold=0.01)  # Prune small weights

    # Save compressed model
    output_path = "checkpoints/compressed_model_5050.pt"
    save_compressed_model(model, output_path)

    print("Model compression completed!")


if __name__ == "__main__":
    main()

import os

import gradio as gr
import torch
from transformers import AutoTokenizer

from model import LlamaForCausalLM
from train import ModelConfig


def load_model(checkpoint_path):
    """Load the model and tokenizer."""
    # Initialize model config
    model_config = ModelConfig()

    # Initialize model
    model = LlamaForCausalLM(model_config)

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

    # Move model to available device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def old_generate_text(prompt, max_length=100, num_sequences=1, temperature=0.7):
    """Generate text based on the prompt."""
    model.eval()  # Ensure model is in eval mode
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)  # Add attention mask like in training

    generated_sequences = []

    with torch.no_grad():
        for _ in range(num_sequences):
            generated_tokens = input_ids[0].tolist()
            current_input_ids = input_ids.clone()
            current_mask = attention_mask.clone()

            for _ in range(max_length):
                # Get model outputs like in training
                outputs = model(current_input_ids, attention_mask=current_mask)
                next_token_logits = outputs[..., -1, :] / temperature

                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append new token
                generated_tokens.append(next_token.item())

                # Update input_ids and attention_mask
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                current_mask = torch.cat(
                    [current_mask, torch.ones_like(next_token)], dim=1
                )

                if next_token.item() == tokenizer.eos_token_id:
                    break

            # Decode the generated sequence
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_sequences.append(decoded_text)

    return "\n\n---\n\n".join(generated_sequences)


def generate_text(prompt, max_length=100):
    model.eval()
    # Get device from model parameters
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        generated_tokens = input_ids[0].tolist()  # Start with input tokens

        for _ in range(max_length):
            # Get model outputs
            outputs = model(input_ids)
            next_token_logits = outputs[..., -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Append new token
            generated_tokens.append(next_token.item())
            # Reshape next_token to match input_ids dimensions
            next_token = next_token.unsqueeze(-1)  # Add sequence length dimension
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# Load model and tokenizer globally
print("Loading model and tokenizer...")
checkpoint_path = "checkpoints/checkpoint_50.pt"  # Adjust path as needed
model, tokenizer, device = load_model(checkpoint_path)
print(f"Model loaded and running on {device}")


# Create Gradio interface
def gradio_interface(prompt, max_length):
    try:
        return generate_text(
            prompt,
            max_length=int(max_length),
        )
    except Exception as e:
        return f"Error: {str(e)}"


# Define the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(
            label="Enter your prompt", placeholder="Once upon a time...", lines=3
        ),
        gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Maximum Length"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="SmolLM2 Text Generator",
    description="""Enter a prompt and adjust the parameters to generate text.
    - Maximum Length: Controls how long the generated text can be
    """,
)

if __name__ == "__main__":
    iface.launch(share=True)

import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model import LlamaForCausalLM

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_logs.txt"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 512
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 0
    eos_token_id: int = 0


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read data directly from input.txt
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.read().split("\n\n")

        # Filter out empty strings
        self.data = [text for text in self.data if text.strip()]
        logger.info(f"Loaded {len(self.data)} text segments from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, step, loss, save_path):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved at step {step}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"], checkpoint["loss"]


def generate_sample_text(model, tokenizer, prompt, max_length=100):
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

    model.train()
    return tokenizer.decode(generated_tokens)


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_size(model):
    """Print model size in millions of parameters."""
    total_params = count_parameters(model)
    logger.info(
        f"\nTotal trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)"
    )


def main():
    # Load configuration
    config = load_config("config_smollm2_135M.yaml")

    # Setup device with MPS support check
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Initialize model config
    model_config = ModelConfig()

    # Initialize model
    model = LlamaForCausalLM(model_config).to(device)

    # Print model architecture and size
    logger.info("\nModel Architecture:")
    logger.info("=" * 50)
    logger.info(model)
    logger.info("=" * 50)
    print_model_size(model)
    logger.info("=" * 50 + "\n")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["tokenizer_name_or_path"]
    )

    # Set padding token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate_scheduler"]["learning_rate"],
        betas=(
            config["optimizer"]["optimizer_factory"]["adam_beta1"],
            config["optimizer"]["optimizer_factory"]["adam_beta2"],
        ),
        eps=config["optimizer"]["optimizer_factory"]["adam_eps"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # Initialize dataset and dataloader
    dataset = TextDataset(
        "input.txt",
        tokenizer,
        max_length=512,  # Reduced from original length
    )

    # Reduce batch size to handle memory constraints
    micro_batch_size = 2  # Reduced from original
    gradient_accumulation_steps = 4  # To maintain effective batch size

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,  # Reduced batch size
        shuffle=True,
        num_workers=config["data_stages"][0]["data"]["num_loading_workers"],
    )

    # Check for existing checkpoint at 5000 steps
    checkpoint_dir = config["checkpoints"]["checkpoints_path"]
    final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_5000.pt")
    start_step = 0

    if os.path.exists(final_checkpoint_path):
        logger.info(
            "Found checkpoint at 5000 steps. Loading and training for 50 more steps."
        )
        start_step, _ = load_checkpoint(model, optimizer, final_checkpoint_path)
        # total_steps = start_step + 50
        total_steps = start_step + 5
    else:
        logger.info("No checkpoint found at 5000 steps. Starting fresh training.")
        # total_steps = 5000
        total_steps = 50

    # Training loop with tqdm
    model.train()
    running_loss = 0
    step = start_step
    accumulated_steps = 0

    test_prompt = "Once upon a time, in a distant galaxy"

    progress_bar = tqdm(total=total_steps, initial=start_step, desc="Training")

    while step < total_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item() * gradient_accumulation_steps
            accumulated_steps += 1

            # Only update weights after accumulating enough gradients
            if accumulated_steps % gradient_accumulation_steps == 0:
                if config["optimizer"]["clip_grad"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["optimizer"]["clip_grad"]
                    )

                optimizer.step()
                optimizer.zero_grad()
                step += 1
                progress_bar.update(1)

                # Update progress bar description with loss
                if step % 10 == 0:
                    progress_bar.set_description(
                        f"Training (loss: {running_loss/10:.4f})"
                    )
                    running_loss = 0

                # if step % 50 == 0:
                if step % 10 == 0:
                    # Clear cache before generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps"):
                        import gc

                        gc.collect()
                        torch.mps.empty_cache()

                    # Generate sample text
                    generated_text = generate_sample_text(model, tokenizer, test_prompt)
                    logger.info(f"\nSample generation at step {step}:")
                    logger.info(f"Prompt: {test_prompt}")
                    logger.info(f"Generated: {generated_text}\n")

                    # Save checkpoint
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        loss.item()
                        * gradient_accumulation_steps,  # Rescale loss for logging
                        os.path.join(checkpoint_dir, f"checkpoint_{step}.pt"),
                    )

                if step >= total_steps:
                    break

            # Memory optimization: Clear memory after each forward/backward pass
            if hasattr(torch.backends, "mps"):
                import gc

                gc.collect()
                torch.mps.empty_cache()

        if step >= total_steps:
            break

    progress_bar.close()

    # Save final checkpoint
    save_checkpoint(
        model,
        optimizer,
        step,
        loss.item(),
        os.path.join(checkpoint_dir, f"checkpoint_{step}.pt"),
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()

"""
Base Model Safety Fine-tuning (Full Parameter FT)

Goal:
- Fair comparison baseline for SN-Tune.
- Keep training setup aligned with sn_tune.py, except:
  - Do NOT load safety neurons
  - Do NOT apply gradient masking
  - Fine-tune all model parameters on the same safety dataset

Usage:
python base_model_Safety_ft.py \
  ./corpus_all/circuit_breakers_train.json \
  ./base_model_safety_tf
"""

import os
import sys
import json
import torch
from bitsandbytes.optim import AdamW8bit
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# Configuration (matched to sn_tune.py)
# =====================================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B"

LEARNING_RATE = 1e-6
NUM_EPOCHS = 3
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 512
MAX_SAMPLES = 4994

OUTPUT_DIR_DEFAULT = "./base_model_safety_tf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# Safety Dataset (same format as sn_tune.py)
# =====================================================================
class SafetyDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_samples=None, max_length=512):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[: min(max_samples, len(self.data))]

        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if idx == 0:
            logger.info("\n[Dataset Sample #0]")
            logger.info(f"  Keys: {item.keys()}")
            logger.info(f"  Prompt (first 100 chars): {item.get('prompt', '')[:100]}...")
            logger.info(f"  Response (first 100 chars): {item.get('llama3_output', '')[:100]}...")

        harmful_prompt = item.get("prompt", "")
        safe_response = item.get("llama3_output", "")

        full_text = f"{harmful_prompt} {safe_response}"

        encodings = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = encodings["input_ids"].clone()
        labels[encodings["attention_mask"] == 0] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


# =====================================================================
# Training Loop (full-parameter baseline)
# =====================================================================
def train_base_safety_ft(
    model,
    train_dataloader,
    learning_rate=1e-6,
    num_epochs=3,
    device=DEVICE,
):
    model = model.to(device)
    model.train()

    optimizer = AdamW8bit(model.parameters(), lr=learning_rate)

    total_loss = 0.0
    total_steps = 0

    logger.info("Starting Base Model Safety FT training...")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Num batches: {len(train_dataloader)}")

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        pbar = tqdm(train_dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if batch_idx == 0:
                logger.info("\n[First Batch Info]")
                logger.info(f"  Batch size: {input_ids.shape[0]}")
                logger.info(f"  Sequence length: {input_ids.shape[1]}")
                logger.info(f"  Device: {input_ids.device}")
                valid_labels = (labels != -100).sum().item()
                logger.info(f"  Valid labels (non-padding): {valid_labels}/{labels.numel()}")

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected at batch {batch_idx + 1}. Skipping this batch.")
                continue

            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            epoch_loss += loss_val
            total_steps += 1

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                avg_batch_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_batch_loss:.4f}"})
                logger.info(f"  Batch {batch_idx + 1}: loss = {loss_val:.4f}")

        logger.info(f"Epoch {epoch + 1} completed - Epoch Loss: {epoch_loss / len(train_dataloader):.4f}")

    avg_loss = total_loss / max(1, total_steps)
    logger.info(f"\n{'=' * 70}")
    logger.info("Training Complete")
    logger.info(f"{'=' * 70}")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Training time: {num_epochs} epoch(s)")
    logger.info(f"{'=' * 70}\n")

    return model


# =====================================================================
# Save Model
# =====================================================================
def save_model_and_tokenizer(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")


# =====================================================================
# Main
# =====================================================================
def main(argv):
    if len(argv) < 1:
        logger.error("Usage: python base_model_Safety_ft.py <safety_dataset_json> [output_dir]")
        logger.error("Example: python base_model_Safety_ft.py ./corpus_all/circuit_breakers_train.json ./base_model_safety_tf")
        sys.exit(1)

    safety_dataset_json = argv[0]
    output_dir = argv[1] if len(argv) > 1 else OUTPUT_DIR_DEFAULT

    if not os.path.exists(safety_dataset_json):
        logger.error(f"Safety dataset file not found: {safety_dataset_json}")
        sys.exit(1)

    logger.info(f"\n{'=' * 70}")
    logger.info("Base Model Safety Fine-tuning (Full Parameter FT)")
    logger.info(f"{'=' * 70}")
    logger.info(f"Base model: {MODEL_NAME}")
    logger.info(f"Safety dataset file: {safety_dataset_json}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training setup: LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, MaxSamples={MAX_SAMPLES}")
    logger.info(f"{'=' * 70}\n")

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    logger.info("✓ Model and tokenizer loaded (float32)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    logger.info("\nLoading safety dataset...")
    safety_dataset = SafetyDataset(
        safety_dataset_json,
        tokenizer,
        max_samples=MAX_SAMPLES,
        max_length=MAX_SEQ_LENGTH,
    )

    train_dataloader = DataLoader(
        safety_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    logger.info(f"✓ DataLoader created: {len(train_dataloader)} batches")
    logger.info(f"  Total samples: {len(safety_dataset)}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Max sequence length: {MAX_SEQ_LENGTH}")

    logger.info("\nStarting base model safety fine-tuning...")
    model = train_base_safety_ft(
        model,
        train_dataloader,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
    )

    logger.info("\nSaving fine-tuned model...")
    save_model_and_tokenizer(model, tokenizer, output_dir)

    logger.info(f"\n{'=' * 70}")
    logger.info("Base Model Safety FT Complete!")
    logger.info(f"{'=' * 70}")
    logger.info(f"Fine-tuned model saved to: {output_dir}")
    logger.info(f"{'=' * 70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])

# ============================
# Path fix (must be first)
# ============================
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================
# Imports
# ============================
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.dataset import ASRDataset

# ============================
# Config
# ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSOR_NAME = "facebook/wav2vec2-base-960h"      # tokenizer source
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"      # acoustic model

CSV_PATH = "data/processed/train.csv"
SAVE_DIR = "models/asr_model"

EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
MAX_STEPS_PER_EPOCH = 50
LEARNING_RATE = 1e-4


# ============================
# Training
# ============================
def train():
    print("🔹 Loading processor (local cache only)...")
    processor = Wav2Vec2Processor.from_pretrained(
        PROCESSOR_NAME,
        local_files_only=True
    )

    print("🔹 Loading XLSR model (local cache only)...")
    model = Wav2Vec2ForCTC.from_pretrained(
    SAVE_DIR,
    pad_token_id=processor.tokenizer.pad_token_id
)


    model.to(DEVICE)

    
    print("🔹 Loading dataset...")
    dataset = ASRDataset(CSV_PATH, processor)

    # ----------------------------
    # Collate function (NOW SAFE)
    # ----------------------------
    def collate_fn(batch):
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        batch_inputs = processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt"
        )

        with processor.as_target_processor():
            batch_labels = processor.pad(
                {"input_ids": labels},
                padding=True,
                return_tensors="pt"
            )

        batch_inputs["labels"] = batch_labels["input_ids"]
        return batch_inputs

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = GradScaler()
    model.train()

    print("🚀 Training started")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            total=MAX_STEPS_PER_EPOCH
        )

        for step, batch in enumerate(progress):
            if step >= MAX_STEPS_PER_EPOCH:
                break

            input_values = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with autocast():
                outputs = model(
                    input_values=input_values,
                    labels=labels
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            progress.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.3f}")

        avg_loss = epoch_loss / MAX_STEPS_PER_EPOCH
        print(f"\n✅ Epoch {epoch+1} finished | Avg loss: {avg_loss:.4f}")

        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)

    print("🎉 Training complete")
    print(f"📦 Model saved to {SAVE_DIR}")


if __name__ == "__main__":
    train()

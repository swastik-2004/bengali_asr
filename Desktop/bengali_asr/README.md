Bengali ASR using Wav2Vec2-XLSR (Exploratory Project)
📌 Project Overview

This project explores building an Automatic Speech Recognition (ASR) system for Bengali (bn) using a pretrained Wav2Vec2-XLSR model from Hugging Face.

The goal was to:

Train a character-level CTC-based ASR model

Handle real-world training issues (padding, CTC loss, blank dominance)

Deploy inference via FastAPI (planned)

While the full end-to-end ASR system did not reach usable inference quality, the project successfully demonstrates model training, debugging, and diagnosis in a low-resource ASR setting.

🧠 What Was Implemented Successfully
✅ Data Pipeline

Used Google FLEURS (bn_in) dataset

Streaming download with sample limiting

Audio preprocessing at 16 kHz mono

CSV-based dataset indexing

✅ Model Training

Base model: facebook/wav2vec2-large-xlsr-53

CTC head fine-tuning using PyTorch

Mixed precision training (AMP)

Gradient accumulation for low VRAM GPUs (RTX 3050, 4GB)

Custom collate_fn for variable-length audio

Training stability fixes:

Encoder freezing

Step-capped epochs

Windows-safe DataLoader configuration

✅ Training Results

Loss reduced consistently across epochs:

Epoch 1 avg loss ≈ 4025
Epoch 2 avg loss ≈ 1439
Epoch 3 avg loss ≈  550


This confirmed:

Correct loss computation

Proper alignment learning

Stable optimization

❌ What Did Not Work (And Why)
Issue: Blank-Dominant Inference (Empty Output)

During inference, the model predicted only the CTC blank token (ID = 0), resulting in empty transcriptions.

Root Causes (Identified & Confirmed)

Tokenizer–Language Mismatch

Used wav2vec2-base-960h tokenizer (English-centric)

Bengali grapheme distribution poorly represented

CTC strongly biased toward predicting blank

Low-Resource Training Regime

~1500 Bengali samples

Very large XLSR model (300M+ parameters)

No external language model (LM)

Character-level CTC without subword modeling

CTC Blank Dominance

Even after encoder unfreezing and fine-tuning

Model converged to blank-only predictions

This is a known failure mode in low-resource ASR

Conclusion

This was not a bug, but a data + tokenizer limitation.

🧪 Debugging & Validation Performed

Verified logits distribution (non-zero, healthy range)

Confirmed model weights loaded correctly

Verified decoding logic (CTC collapse, blank removal)

Tested unfreezing encoder with reduced learning rate

Confirmed token IDs remained all-zero during inference

This conclusively ruled out:

Inference bugs

Training loop bugs

Model loading errors

📌 Key Learnings

XLSR fine-tuning requires a language-appropriate tokenizer

Low-resource ASR needs:

More data, or

Subword tokenization, or

External language models

Loss reduction alone does not guarantee usable ASR

CTC blank dominance is a common real-world failure mode

🚀 Next Steps

This repository is intentionally archived at this stage.

The next phase of this work will:

Switch to a high-resource language (Hindi)

Use a tokenizer and dataset aligned with the language

Reuse the same training and deployment pipeline

Achieve working ASR inference and API deployment

👉 See upcoming repository: Hindi ASR with Wav2Vec2 + FastAPI

🛠️ Tech Stack

Python

PyTorch

Hugging Face Transformers

Wav2Vec2 / XLSR

Datasets (Google FLEURS)

FastAPI (planned)

NVIDIA GPU (RTX 3050, 4GB)

🧑‍💻 Author Notes

This project emphasizes engineering honesty:

Not all ML projects succeed

Correct diagnosis is as valuable as a working model

The same pipeline, applied to the right data, does succeed

import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models/asr_model"

print("🔹 Loading ASR model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()
print("✅ Model loaded")


# ----------------------------
# Transcription
# ----------------------------
def transcribe_audio(audio_path: str) -> str:
    audio, sr = sf.read(audio_path)

    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE)).logits
        probs = torch.softmax(logits, dim=-1)[0]  # (time, vocab)

    blank_id = processor.tokenizer.pad_token_id  # 0
    MIN_CONFIDENCE = 0.15  # very important

    raw_tokens = []

    # ----------------------------
    # Token selection
    # ----------------------------
    for t in range(probs.size(0)):
        sorted_probs, sorted_ids = probs[t].sort(descending=True)

        top_prob = sorted_probs[0].item()
        top_id = sorted_ids[0].item()
        second_id = sorted_ids[1].item()

        # If confident and not blank → take it
        if top_id != blank_id and top_prob > MIN_CONFIDENCE:
            raw_tokens.append(top_id)

        # If blank dominates → try 2nd-best
        elif second_id != blank_id:
            raw_tokens.append(second_id)

        # else: skip timestep completely

    # ----------------------------
    # TRUE CTC collapse
    # ----------------------------
    collapsed = []
    prev = None

    for tok in raw_tokens:
        if tok != prev:
            collapsed.append(tok)
        prev = tok

    # Remove blanks if any survived
    collapsed = [t for t in collapsed if t != blank_id]

    if not collapsed:
        return "[no confident transcription]"

    predicted_ids = torch.tensor(collapsed)

    transcription = processor.tokenizer.decode(
        predicted_ids,
        skip_special_tokens=True
    )

    return transcription.strip()

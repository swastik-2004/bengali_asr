import os
import re
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ----------------------------
# Resolve project root (notebook-safe)
# ----------------------------
PROJECT_ROOT = os.getcwd()
if not os.path.exists(os.path.join(PROJECT_ROOT, "src")):
    PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

print("📁 PROJECT ROOT:", PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_AUDIO_DIR = os.path.join(DATA_DIR, "processed", "audio")
CSV_PATH = os.path.join(DATA_DIR, "processed", "train.csv")

os.makedirs(RAW_AUDIO_DIR, exist_ok=True)


# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ----------------------------
# Preprocess
# ----------------------------
def preprocess(dataset):
    records = []

    print("🔹 Starting preprocessing")
    print(f"Total samples received: {len(dataset)}")
    print("📂 Writing audio to:", RAW_AUDIO_DIR)
    print("🧾 Writing CSV to:", CSV_PATH)

    for idx, sample in enumerate(tqdm(dataset, desc="Saving audio + text")):
        try:
            # -------- TEXT --------
            text = sample.get("transcription") or sample.get("sentence")
            if not text:
                continue

            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue

            # -------- AUDIO (ROBUST) --------
            audio = sample["audio"]

            if isinstance(audio, dict):
                audio_array = audio["array"]
                sampling_rate = audio["sampling_rate"]
            else:
                audio_array = audio
                sampling_rate = 16000  # FLEURS default

            audio_path = os.path.join(RAW_AUDIO_DIR, f"sample_{idx}.wav")

            sf.write(audio_path, audio_array, sampling_rate)

            records.append({
                "audio_path": audio_path,
                "text": cleaned_text
            })

        except Exception as e:
            continue  # skip bad samples safely

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print("✅ Preprocessing complete")
    print(f"Saved {len(df)} samples")

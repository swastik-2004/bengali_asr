import json
import pandas as pd
from collections import Counter


CSV_PATH = "data/processed/train.csv"
VOCAB_PATH = "models/bengali_vocab.json"


def build_vocab():
    df = pd.read_csv(CSV_PATH)

    all_text = " ".join(df["text"].tolist())

    # Count characters
    vocab_counter = Counter(all_text)

    # Sort characters (for reproducibility)
    vocab_list = sorted(vocab_counter.keys())

    # CTC required tokens
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    for char in vocab_list:
        if char not in vocab:
            vocab[char] = len(vocab)

    # Save vocab
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✅ Vocabulary built with {len(vocab)} tokens")
    print(f"Saved to {VOCAB_PATH}")


if __name__ == "__main__":
    build_vocab()

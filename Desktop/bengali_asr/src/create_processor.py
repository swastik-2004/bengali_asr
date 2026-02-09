from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)


VOCAB_PATH = "models/bengali_vocab.json"
PROCESSOR_DIR = "models/bengali_processor"


def create_processor():
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=VOCAB_PATH,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token=" "
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    processor.save_pretrained(PROCESSOR_DIR)

    print("✅ Bengali processor created and saved")
    print(f"Saved to {PROCESSOR_DIR}")


if __name__ == "__main__":
    create_processor()

import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "models/asr_model"
AUDIO_PATH = "C:\\Users\\swastik dasgupta\\Desktop\\bengali_asr\\data\\processed\\audio\\sample_12.wav"

processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

speech, sr = sf.read(AUDIO_PATH)

inputs = processor(
    speech,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    logits = model(inputs.input_values.to(DEVICE)).logits

print("Logits max:", logits.max().item())
print("Logits mean:", logits.mean().item())

pred_ids = torch.argmax(logits, dim=-1)
print("Raw token IDs:", pred_ids)
print("Decoded:", processor.batch_decode(pred_ids))


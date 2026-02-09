import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException

from api.inference import transcribe_audio
from api.schemas import TranscriptionResponse


app = FastAPI(
    title="Bengali ASR API",
    description="Speech to Text using wav2vec2 + XLSR",
    version="1.0"
)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    temp_file = f"temp_{uuid.uuid4().hex}.wav"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = transcribe_audio(temp_file)
    finally:
        os.remove(temp_file)

    return {"text": text}


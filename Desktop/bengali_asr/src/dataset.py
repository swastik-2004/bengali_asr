import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset

class ASRDataset(Dataset):
    def __init__(self,csv_path,processor):
        self.df=pd.read_csv(csv_path)
        self.processor=processor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row=self.df.iloc[idx]

        audio,sr=sf.read(row["audio_path"])

        inputs=self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with self.processor.as_target_processor():
            labels=self.processor(
                row["text"],
                return_tensors="pt",
                padding=True
            )

        return {
            "input_values":inputs.input_values.squeeze(0),
            "labels":labels.input_ids.squeeze(0)}

    
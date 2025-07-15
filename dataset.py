# Built-in
from pathlib import Path

# NN Libraries
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # get training data or test data based on condition if fold is 5
        if split == 'train': self.metadata = self.metadata[self.metadata['fold'] != 5]
        else: self.metadata = self.metadata[self.metadata['fold'] == 5]

        # categories the data
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self): return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform: spectrogram = self.transform(waveform)
        else: spectrogram = waveform

        return spectrogram, row['label']

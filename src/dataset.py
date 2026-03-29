"""Dataset classes for BirdCLEF 2026."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Any

from .preprocessing import load_audio, load_audio_segment, audio_to_melspectrogram


class BirdCLEFDataset(Dataset):
    """
    Dataset for BirdCLEF training data (train_audio).
    """
    
    def __init__(
        self,
        df,
        audio_dir: Path,
        species_to_idx: dict,
        cfg,
        augmentation=None,
        is_train: bool = True
    ):
        """
        Args:
            df: DataFrame with 'filename' and 'target' columns
            audio_dir: Directory containing audio files
            species_to_idx: Dictionary mapping species names to indices
            cfg: Configuration object with audio parameters
            augmentation: Audio augmentation pipeline (audiomentations)
            is_train: Whether this is training data
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.species_to_idx = species_to_idx
        self.cfg = cfg
        self.augmentation = augmentation
        self.is_train = is_train
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load audio
        filepath = self.audio_dir / row['filename']
        audio = load_audio(
            str(filepath),
            duration=self.cfg.audio.duration,
            sr=self.cfg.audio.sample_rate
        )
        
        # Apply augmentation
        if self.is_train and self.augmentation is not None:
            audio = self.augmentation(samples=audio, sample_rate=self.cfg.audio.sample_rate)
        
        # Convert to mel-spectrogram
        mel_spec = audio_to_melspectrogram(
            audio,
            sr=self.cfg.audio.sample_rate,
            n_mels=self.cfg.audio.n_mels,
            fmin=self.cfg.audio.fmin,
            fmax=self.cfg.audio.fmax,
            n_fft=self.cfg.audio.n_fft,
            hop_length=self.cfg.audio.hop_length
        )
        
        # Convert to 3-channel image (duplicate channels for pretrained models)
        mel_spec = np.stack([mel_spec, mel_spec, mel_spec], axis=0)
        
        # Get target
        target = row['target']
        
        return {
            'image': torch.tensor(mel_spec, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }


class TestDataset(Dataset):
    """
    Dataset for BirdCLEF test soundscapes.
    """
    
    def __init__(self, df, audio_dir: Path, cfg):
        """
        Args:
            df: DataFrame with 'row_id', 'filename', and 'start_time' columns
            audio_dir: Directory containing test soundscape files
            cfg: Configuration object with audio parameters
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.cfg = cfg
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Load audio segment
        filepath = self.audio_dir / row['filename']
        audio = load_audio_segment(
            str(filepath),
            row['start_time'],
            duration=self.cfg.audio.duration,
            sr=self.cfg.audio.sample_rate
        )
        
        # Convert to mel-spectrogram
        mel_spec = audio_to_melspectrogram(
            audio,
            sr=self.cfg.audio.sample_rate,
            n_mels=self.cfg.audio.n_mels,
            fmin=self.cfg.audio.fmin,
            fmax=self.cfg.audio.fmax,
            n_fft=self.cfg.audio.n_fft,
            hop_length=self.cfg.audio.hop_length
        )
        
        # Convert to 3-channel image
        mel_spec = np.stack([mel_spec, mel_spec, mel_spec], axis=0)
        
        return {
            'image': torch.tensor(mel_spec, dtype=torch.float32),
            'row_id': row['row_id']
        }


class TrainSoundscapesDataset(Dataset):
    """
    Dataset for labeled train soundscapes.
    """
    
    def __init__(
        self,
        df,
        audio_dir: Path,
        species_to_idx: dict,
        cfg,
        augmentation=None,
        is_train: bool = True
    ):
        """
        Args:
            df: DataFrame with 'filename', 'start', 'end', and 'primary_label' columns
            audio_dir: Directory containing soundscape files
            species_to_idx: Dictionary mapping species names to indices
            cfg: Configuration object with audio parameters
            augmentation: Audio augmentation pipeline
            is_train: Whether this is training data
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.species_to_idx = species_to_idx
        self.cfg = cfg
        self.augmentation = augmentation
        self.is_train = is_train
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load audio segment
        filepath = self.audio_dir / row['filename']
        audio = load_audio_segment(
            str(filepath),
            row['start'],
            duration=self.cfg.audio.duration,
            sr=self.cfg.audio.sample_rate
        )
        
        # Apply augmentation
        if self.is_train and self.augmentation is not None:
            audio = self.augmentation(samples=audio, sample_rate=self.cfg.audio.sample_rate)
        
        # Convert to mel-spectrogram
        mel_spec = audio_to_melspectrogram(
            audio,
            sr=self.cfg.audio.sample_rate,
            n_mels=self.cfg.audio.n_mels,
            fmin=self.cfg.audio.fmin,
            fmax=self.cfg.audio.fmax,
            n_fft=self.cfg.audio.n_fft,
            hop_length=self.cfg.audio.hop_length
        )
        
        # Convert to 3-channel image
        mel_spec = np.stack([mel_spec, mel_spec, mel_spec], axis=0)
        
        # Create target from semicolon-separated labels
        target = np.zeros(len(self.species_to_idx), dtype=np.float32)
        labels = str(row['primary_label']).split(';')
        for label in labels:
            label = label.strip()
            if label in self.species_to_idx:
                target[self.species_to_idx[label]] = 1.0
        
        return {
            'image': torch.tensor(mel_spec, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }

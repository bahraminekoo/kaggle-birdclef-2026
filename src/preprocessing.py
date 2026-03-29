"""Audio preprocessing utilities for BirdCLEF 2026."""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple


def load_audio(filepath: str, duration: int = 5, sr: int = 32000) -> np.ndarray:
    """
    Load audio file and ensure it's the correct duration.
    
    Args:
        filepath: Path to audio file
        duration: Target duration in seconds
        sr: Target sample rate
        
    Returns:
        Audio array of shape (sr * duration,)
    """
    try:
        # Load audio
        audio, orig_sr = sf.read(filepath)
        
        # Resample if needed
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Ensure correct duration
        target_length = sr * duration
        
        if len(audio) > target_length:
            # Random crop
            start = np.random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        return audio.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        # Return silence if error
        return np.zeros(sr * duration, dtype=np.float32)


def load_audio_segment(
    filepath: str, 
    start_time: float, 
    duration: int = 5, 
    sr: int = 32000
) -> np.ndarray:
    """
    Load a specific segment from an audio file.
    
    Args:
        filepath: Path to audio file
        start_time: Start time in seconds
        duration: Segment duration in seconds
        sr: Target sample rate
        
    Returns:
        Audio array of shape (sr * duration,)
    """
    try:
        # Calculate offset in samples
        offset = int(start_time * sr)
        num_samples = int(duration * sr)
        
        # Load audio segment
        audio, orig_sr = sf.read(filepath, start=offset, frames=num_samples)
        
        # Resample if needed
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Ensure correct length
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]
        
        return audio.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading {filepath} at {start_time}s: {e}")
        return np.zeros(sr * duration, dtype=np.float32)


def audio_to_melspectrogram(
    audio: np.ndarray,
    sr: int = 32000,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Convert audio to mel-spectrogram.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Normalized mel-spectrogram of shape (n_mels, time_steps)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm.astype(np.float32)


def create_multilabel_target(row, species_to_idx: dict) -> np.ndarray:
    """
    Create multi-label target vector from dataframe row.
    
    Args:
        row: Pandas dataframe row with 'primary_label' and 'secondary_labels'
        species_to_idx: Dictionary mapping species names to indices
        
    Returns:
        Binary target vector of shape (num_classes,)
    """
    import pandas as pd
    
    target = np.zeros(len(species_to_idx), dtype=np.float32)
    
    # Primary label
    if row['primary_label'] in species_to_idx:
        target[species_to_idx[row['primary_label']]] = 1.0
    
    # Secondary labels
    if pd.notna(row['secondary_labels']):
        secondary = str(row['secondary_labels']).strip('[]').replace("'", "").split(', ')
        for label in secondary:
            label = label.strip()
            if label and label in species_to_idx:
                target[species_to_idx[label]] = 1.0
    
    return target

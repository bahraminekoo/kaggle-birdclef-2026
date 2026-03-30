"""Utility functions for BirdCLEF 2026."""

import numpy as np
import pandas as pd
from pathlib import Path


def load_train_soundscapes_data(soundscapes_dir: Path, labels_csv: Path, species_to_idx: dict):
    """
    Load and prepare train soundscapes data.
    
    Args:
        soundscapes_dir: Path to train_soundscapes directory
        labels_csv: Path to train_soundscapes_labels.csv
        species_to_idx: Dictionary mapping species to indices
        
    Returns:
        DataFrame with filename, start, end, and target columns
    """
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    
    print(f"Loaded {len(labels_df)} labeled soundscape segments")
    
    # Create targets from semicolon-separated labels
    def create_soundscape_target(primary_label_str):
        target = np.zeros(len(species_to_idx), dtype=np.float32)
        labels = str(primary_label_str).split(';')
        for label in labels:
            label = label.strip()
            if label in species_to_idx:
                target[species_to_idx[label]] = 1.0
        return target
    
    labels_df['target'] = labels_df['primary_label'].apply(create_soundscape_target)
    
    # Calculate duration for each segment
    labels_df['duration'] = labels_df['end'] - labels_df['start']
    
    print(f"Average segment duration: {labels_df['duration'].mean():.2f}s")
    print(f"Unique soundscape files: {labels_df['filename'].nunique()}")
    
    return labels_df


def combine_train_data(train_audio_df, train_soundscapes_df, balance_ratio=1.0):
    """
    Combine train_audio and train_soundscapes data.
    
    Args:
        train_audio_df: DataFrame with train_audio data
        train_soundscapes_df: DataFrame with train_soundscapes data
        balance_ratio: Ratio of soundscapes to audio samples (1.0 = equal)
        
    Returns:
        Combined DataFrame with 'source' column indicating origin
    """
    # Add source column
    train_audio_df = train_audio_df.copy()
    train_audio_df['source'] = 'audio'
    
    train_soundscapes_df = train_soundscapes_df.copy()
    train_soundscapes_df['source'] = 'soundscape'
    
    # Balance if needed
    if balance_ratio < 1.0:
        n_soundscapes = int(len(train_audio_df) * balance_ratio)
        train_soundscapes_df = train_soundscapes_df.sample(n=min(n_soundscapes, len(train_soundscapes_df)), random_state=42)
    
    # Combine
    combined_df = pd.concat([train_audio_df, train_soundscapes_df], ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Train audio samples: {(combined_df['source'] == 'audio').sum()}")
    print(f"  Train soundscapes: {(combined_df['source'] == 'soundscape').sum()}")
    print(f"  Total: {len(combined_df)}")
    
    return combined_df


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

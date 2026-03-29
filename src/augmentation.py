"""Audio augmentation utilities for BirdCLEF 2026."""

import numpy as np
import audiomentations as AA
import torch


def get_train_augmentation():
    """
    Get audio augmentation pipeline for training.
    
    Returns:
        Audiomentations Compose object
    """
    return AA.Compose([
        AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        AA.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        AA.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        AA.Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
    ])


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.
    
    Applies frequency and time masking to spectrograms.
    """
    
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 30, n_freq_masks: int = 1, n_time_masks: int = 1):
        """
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to a spectrogram.
        
        Args:
            spec: Spectrogram of shape (n_mels, time_steps) or (channels, n_mels, time_steps)
            
        Returns:
            Augmented spectrogram
        """
        spec = spec.copy()
        
        # Handle both 2D and 3D inputs
        if len(spec.shape) == 3:
            n_channels, n_mels, time_steps = spec.shape
        else:
            n_mels, time_steps = spec.shape
            n_channels = 1
            spec = spec[np.newaxis, ...]
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, n_mels - f)
            spec[:, f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, time_steps - t)
            spec[:, :, t0:t0 + t] = 0
        
        # Return original shape
        if n_channels == 1:
            spec = spec[0]
        
        return spec


class Mixup:
    """
    Mixup: Beyond Empirical Risk Minimization.
    
    Mixes two samples and their labels.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Mixup interpolation strength
        """
        self.alpha = alpha
    
    def __call__(self, batch_images: torch.Tensor, batch_targets: torch.Tensor):
        """
        Apply mixup to a batch.
        
        Args:
            batch_images: Batch of images (B, C, H, W)
            batch_targets: Batch of targets (B, num_classes)
            
        Returns:
            Mixed images and targets
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        mixed_targets = lam * batch_targets + (1 - lam) * batch_targets[index]
        
        return mixed_images, mixed_targets

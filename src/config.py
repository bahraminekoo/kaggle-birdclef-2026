"""Configuration classes for BirdCLEF 2026 training and inference."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 32000
    duration: int = 5  # seconds
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 16000
    n_fft: int = 2048
    hop_length: int = 512


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = 'efficientnet_b0'
    pretrained: bool = True
    num_classes: int = 234
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration."""
    n_folds: int = 5
    train_folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-6
    num_workers: int = 2
    device: str = 'cuda'
    
    # Augmentation
    use_augmentation: bool = True
    
    # Checkpoint settings
    save_checkpoint_every_epoch: bool = True
    auto_resume: bool = True


@dataclass
class PathConfig:
    """Path configuration for Kaggle environment."""
    data_dir: Path = Path('/kaggle/input/birdclef-2026')
    train_audio_dir: Path = field(init=False)
    train_soundscapes_dir: Path = field(init=False)
    test_soundscapes_dir: Path = field(init=False)
    output_dir: Path = Path('/kaggle/working')
    model_dir: Path = Path('/kaggle/input/birdclef-2026-trained-models')
    
    def __post_init__(self):
        self.train_audio_dir = self.data_dir / 'train_audio'
        self.train_soundscapes_dir = self.data_dir / 'train_soundscapes'
        self.test_soundscapes_dir = self.data_dir / 'test_soundscapes'


@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 16
    num_workers: int = 0
    device: str = 'cpu'
    model_folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3])


class BaselineConfig:
    """Complete baseline configuration."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
    
    @property
    def data_dir(self):
        return self.paths.data_dir
    
    @property
    def train_audio_dir(self):
        return self.paths.train_audio_dir
    
    @property
    def train_soundscapes_dir(self):
        return self.paths.train_soundscapes_dir
    
    @property
    def output_dir(self):
        return self.paths.output_dir


class ImprovedConfig(BaselineConfig):
    """Improved configuration with Phase 1 enhancements."""
    
    def __init__(self):
        super().__init__()
        # Improved settings
        self.training.epochs = 20
        self.training.lr = 5e-4
        self.audio.n_mels = 224
        
        # Use train soundscapes
        self.use_train_soundscapes = True
        
        # SpecAugment parameters
        self.spec_augment = True
        self.freq_mask_param = 15
        self.time_mask_param = 30
        
        # Mixup
        self.mixup = True
        self.mixup_alpha = 0.3

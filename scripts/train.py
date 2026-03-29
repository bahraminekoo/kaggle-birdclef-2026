"""
Command-line training script for BirdCLEF 2026.

Usage:
    python scripts/train.py --config configs/baseline_config.yaml
    python scripts/train.py --config configs/improved_config.yaml --fold 0
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BaselineConfig
from src.models import create_model
from src.dataset import BirdCLEFDataset
from src.training import train_one_epoch, validate_one_epoch


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # TODO: Convert YAML dict to config object
    # For now, use default config
    return BaselineConfig()


def main():
    parser = argparse.ArgumentParser(description='Train BirdCLEF 2026 model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--fold', type=int, default=None, help='Specific fold to train (optional)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    print(f"Training with config: {args.config}")
    print(f"Model: {cfg.model.model_name}")
    print(f"Epochs: {cfg.training.epochs}")
    
    # TODO: Implement training loop
    # This is a placeholder for future implementation
    print("\nTraining script is under development.")
    print("Please use the Jupyter notebooks for now:")
    print("  - notebooks/01_baseline_training.ipynb")
    print("  - notebooks/03_training_with_checkpoints.ipynb")


if __name__ == '__main__':
    main()

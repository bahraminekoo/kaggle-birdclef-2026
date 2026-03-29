"""
Command-line inference script for BirdCLEF 2026.

Usage:
    python scripts/inference.py --models models/ --output submission.csv
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BaselineConfig
from src.models import create_model
from src.inference import predict


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for BirdCLEF 2026')
    parser.add_argument('--models', type=str, required=True, help='Path to models directory')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output file path')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    
    args = parser.parse_args()
    
    print(f"Loading models from: {args.models}")
    print(f"Output file: {args.output}")
    
    # TODO: Implement inference
    # This is a placeholder for future implementation
    print("\nInference script is under development.")
    print("Please use the Jupyter notebook for now:")
    print("  - notebooks/02_baseline_inference.ipynb")


if __name__ == '__main__':
    main()

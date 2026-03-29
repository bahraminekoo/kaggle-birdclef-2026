"""
Data preparation script for BirdCLEF 2026.

Downloads and organizes competition data.

Usage:
    python scripts/prepare_data.py --download
    python scripts/prepare_data.py --verify
"""

import argparse
from pathlib import Path
import subprocess


def download_data():
    """Download competition data using Kaggle API."""
    print("Downloading BirdCLEF 2026 data...")
    
    try:
        # Download competition data
        subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', 'birdclef-2026'
        ], check=True)
        
        print("✓ Download complete!")
        print("\nNext steps:")
        print("1. Extract the zip file to data/raw/")
        print("2. Run: python scripts/prepare_data.py --verify")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading data: {e}")
        print("\nMake sure you have:")
        print("1. Installed Kaggle API: pip install kaggle")
        print("2. Setup credentials: ~/.kaggle/kaggle.json")
        print("3. Accepted competition rules on Kaggle website")
    except FileNotFoundError:
        print("✗ Kaggle CLI not found!")
        print("Install it with: pip install kaggle")


def verify_data():
    """Verify that all required data files exist."""
    print("Verifying data files...")
    
    data_dir = Path('data/raw')
    
    required_files = [
        'train.csv',
        'taxonomy.csv',
        'sample_submission.csv',
        'train_soundscapes_labels.csv',
        'recording_location.txt'
    ]
    
    required_dirs = [
        'train_audio',
        'train_soundscapes'
    ]
    
    all_good = True
    
    # Check files
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_good = False
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            num_files = len(list(dir_path.glob('*')))
            print(f"✓ {dir_name}/ ({num_files} files)")
        else:
            print(f"✗ {dir_name}/ - MISSING")
            all_good = False
    
    if all_good:
        print("\n✓ All data files verified!")
    else:
        print("\n✗ Some data files are missing.")
        print("Please download and extract the competition data to data/raw/")


def main():
    parser = argparse.ArgumentParser(description='Prepare BirdCLEF 2026 data')
    parser.add_argument('--download', action='store_true', help='Download competition data')
    parser.add_argument('--verify', action='store_true', help='Verify data files')
    
    args = parser.parse_args()
    
    if args.download:
        download_data()
    elif args.verify:
        verify_data()
    else:
        print("Usage:")
        print("  python scripts/prepare_data.py --download  # Download data")
        print("  python scripts/prepare_data.py --verify    # Verify data")


if __name__ == '__main__':
    main()

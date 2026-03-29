# Data Directory

This directory contains the BirdCLEF 2026 competition data.

## Download Instructions

### Option 1: Using Kaggle API (Recommended)

1. Install Kaggle API:
```bash
pip install kaggle
```

2. Setup Kaggle credentials:
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. Download competition data:
```bash
kaggle competitions download -c birdclef-2026
```

4. Extract to this directory:
```bash
unzip birdclef-2026.zip -d data/raw/
```

### Option 2: Manual Download

1. Go to https://www.kaggle.com/competitions/birdclef-2026/data
2. Download all data files
3. Extract to `data/raw/`

## Directory Structure

```
data/
├── README.md                    # This file
├── raw/                         # Original competition data (gitignored)
│   ├── train_audio/            # Training audio files (~16GB)
│   ├── train_soundscapes/      # Training soundscapes
│   ├── test_soundscapes/       # Test soundscapes (populated on submission)
│   ├── train.csv               # Training metadata
│   ├── train_soundscapes_labels.csv  # Labeled soundscape segments
│   ├── taxonomy.csv            # Species information
│   ├── sample_submission.csv   # Submission format
│   └── recording_location.txt  # Recording location info
│
├── processed/                   # Preprocessed features (gitignored)
│   └── spectrograms/           # Cached spectrograms (optional)
│
└── external/                    # External datasets (gitignored)
    └── pretrained/             # Pretrained model weights
```

## Data Files

### train_audio/
- Short recordings of individual species
- Source: xeno-canto.org and iNaturalist
- Format: OGG, 32 kHz
- Size: ~16 GB
- Files: ~40,000+ recordings

### train_soundscapes/
- 1-minute soundscape recordings
- Same locations as test data
- Some segments labeled by experts
- Format: OGG, 32 kHz

### test_soundscapes/
- Hidden test set (populated during submission)
- ~600 files, 1-minute each
- Format: OGG, 32 kHz

### Metadata Files

**train.csv** - Training metadata
- primary_label: Species code
- secondary_labels: Additional species
- latitude, longitude: Recording location
- author: Recordist name
- filename: Audio filename
- rating: Quality rating (1-5)
- collection: XC or iNat

**train_soundscapes_labels.csv** - Labeled soundscape segments
- filename: Soundscape file
- start, end: Segment timestamps
- primary_label: Semicolon-separated species codes

**taxonomy.csv** - Species information
- 234 species/classes
- primary_label: Species code
- scientific_name: Scientific name
- common_name: Common name
- taxon_class: Aves, Amphibia, Mammalia, Insecta, Reptilia

## Data Statistics

- **Total species:** 234
- **Training samples:** ~40,000+
- **Labeled soundscapes:** ~1,000+ segments
- **Test soundscapes:** ~600 files
- **Audio duration:** 5 seconds per segment
- **Sample rate:** 32,000 Hz

## Important Notes

⚠️ **Large Files:** The data directory is gitignored due to size (>16GB)

⚠️ **Test Data:** Test soundscapes are only available during notebook submission

✅ **Train Soundscapes:** Critical for performance - same domain as test data

## Preprocessing (Optional)

To speed up training, you can preprocess spectrograms:

```python
# Example preprocessing script
from src.preprocessing import audio_to_melspectrogram
import numpy as np

# Cache spectrograms to data/processed/
# This can reduce training time significantly
```

## External Data

You can add external data to `data/external/`:
- Pretrained model weights
- Additional audio datasets
- Metadata from eBird/iNaturalist

**Note:** Only freely available data is allowed per competition rules.

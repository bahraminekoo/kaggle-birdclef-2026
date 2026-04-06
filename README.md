# BirdCLEF 2026 - Pantanal Wildlife Audio Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)

A machine learning framework for identifying wildlife species from audio recordings in Brazil's Pantanal wetlands. This project was developed for the [BirdCLEF 2026 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2026).

## 🎯 Competition Overview

**Goal:** Identify 234 wildlife species (birds, amphibians, mammals, reptiles, insects) from 5-second audio segments.

**Evaluation:** Macro-averaged ROC-AUC (skipping classes with no true positives)

**Current Score:** 0.857 (Public LB) - Rank 1123/1854

**Target:** 0.90+ (Top 15-20%)

## 📁 Project Structure

```
BirdClef2026/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── notebooks/                   # Jupyter notebooks
│   ├── 02_baseline_inference.ipynb
│   ├── 03_training_with_checkpoints.ipynb
│   └── 04_phase1_improved_training.ipynb
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── config.py               # Configuration classes
│   ├── dataset.py              # Dataset classes
│   ├── models.py               # Model definitions
│   ├── augmentation.py         # Audio augmentation
│   ├── preprocessing.py        # Audio preprocessing
│   ├── training.py             # Training utilities
│   └── inference.py            # Inference utilities
│
├── docs/                        # Documentation
│   ├── CHECKPOINT_GUIDE.md     # Checkpoint system guide
│   ├── IMPROVEMENT_PLAN.md     # Score improvement roadmap
│   ├── PHASE1_IMPLEMENTATION.md # Phase 1 implementation guide
│   ├── COMPETITION_OVERVIEW.md # Competition details
│   └── RESULTS.md              # Experiment tracking
│
├── configs/                     # Configuration files
├── scripts/                     # Utility scripts
├── data/                        # Data directory (gitignored)
├── models/                      # Saved models (gitignored)
├── outputs/                     # Training outputs (gitignored)
└── figures/                     # Visualizations
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bahraminekoo/kaggle-birdclef-2026.git
cd kaggle-birdclef-2026

# Install dependencies
pip install -r requirements.txt
```

### Download Data

1. Install Kaggle API:
```bash
pip install kaggle
```

2. Setup Kaggle credentials:
```bash
# Place kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. Download competition data:
```bash
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip -d data/raw/
```

### Training

**Option 1: Using Notebooks (Recommended for Kaggle)**

**Baseline Model:**
1. Open `notebooks/03_training_with_checkpoints.ipynb`
2. Update paths in configuration
3. Run all cells
4. Models will be saved to `outputs/`

**Phase 1 Improved Model:**
1. Open `notebooks/04_phase1_improved_training.ipynb`
2. Run all cells (includes all Phase 1 improvements)
3. Models will be saved to `outputs/`

**Option 2: Using Python Scripts (Coming Soon)**

```bash
python scripts/train.py --config configs/baseline_config.yaml
```

### Inference

1. Open `notebooks/02_baseline_inference.ipynb`
2. Update model paths
3. Run all cells
4. Submission file will be generated as `submission.csv`

## 🏗️ Model Architecture

### Baseline Model

```
Audio (32kHz, 5s) 
    ↓
Mel-Spectrogram (128 x 313)
    ↓
EfficientNet-B0 Backbone
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Linear Layer (234 classes)
    ↓
Sigmoid Activation
    ↓
Multi-label Predictions
```

### Phase 1 Improved Model

```
Audio (32kHz, 5s) - train_audio + train_soundscapes
    ↓
Audio Augmentation (Noise, TimeStretch, PitchShift, Shift)
    ↓
Mel-Spectrogram (224 x 313) - Higher resolution
    ↓
SpecAugment (Frequency & Time Masking)
    ↓
tf_efficientnet_b0_ns Backbone - Better pretrained weights
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Linear Layer (234 classes)
    ↓
Mixup Augmentation (alpha=0.3)
    ↓
Sigmoid Activation
    ↓
Multi-label Predictions
```

### Audio Processing

**Baseline:**
- **Sample Rate:** 32,000 Hz
- **Duration:** 5 seconds
- **Mel-Spectrogram:**
  - n_mels: 128
  - fmin: 20 Hz
  - fmax: 16,000 Hz
  - n_fft: 2048
  - hop_length: 512

**Phase 1 Improved:**
- **Sample Rate:** 32,000 Hz
- **Duration:** 5 seconds
- **Mel-Spectrogram:**
  - n_mels: **224** (increased from 128)
  - fmin: 20 Hz
  - fmax: 16,000 Hz
  - n_fft: 2048
  - hop_length: 512

### Augmentation

**Audio Augmentation:**
- Gaussian Noise (p=0.5)
- Time Stretch (0.8-1.2x, p=0.5)
- Pitch Shift (±2 semitones, p=0.5)
- Time Shift (±0.5s, p=0.5)

**Phase 1 Additional Augmentation:**
- **SpecAugment:** Frequency masking (15 bins) + Time masking (30 frames)
- **Mixup:** Sample mixing with alpha=0.3

## 📊 Results

### Baseline Performance

| Metric | Score |
|--------|-------|
| Local CV | ~0.65-0.75 |
| Public LB | 0.802 |
| Rank | 1077/1529 |

### Phase 1 Results ✅

**Implemented Improvements:**
- ✅ Train soundscapes data integration (critical domain matching)
- ✅ SpecAugment (frequency & time masking)
- ✅ Mixup augmentation (alpha=0.3)
- ✅ Better pretrained model (tf_efficientnet_b0_ns)
- ✅ Higher resolution spectrograms (n_mels=224)
- ✅ Optimized training (12 epochs, batch_size=32)
- ✅ Learning rate (5e-4 with warmup)

**Actual Results:**

| Metric | Baseline | Phase 1 (Partial) | Phase 1 (Latest) | Improvement |
|--------|----------|-------------------|------------------|-------------|
| Local CV (Fold 0) | 0.65-0.75 | **0.94** | **0.94** | +0.19-0.29 |
| Public LB | 0.802 | 0.838 | **0.857** | **+0.055** |
| Rank | 1077/1529 | 1023/1604 | **1123/1854** | Improving |

**Current Status:** 
- ✅ Fold 0 complete (12/12 epochs, CV: 0.94)
- 🔄 Fold 1 in progress (4/12 epochs)
- ⏳ Fold 2 pending
- **Latest submission:** 2-fold partial ensemble (Fold 0 + Fold 1 @ 4 epochs) → **0.857 LB** 🎉

**Training Timeline (30 hrs/week GPU):**
- Week 1: Fold 0 + Fold 1 partial
- Week 2: Fold 1 complete + Fold 2 complete
- Total: 3-fold ensemble

### Improvement Roadmap

See [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) and [`docs/PHASE1_IMPLEMENTATION.md`](docs/PHASE1_IMPLEMENTATION.md) for detailed strategy.

**Phase 2 (Model Improvements):** +0.03-0.05 improvement
- Upgrade to larger model (EfficientNet-B1/B2)
- Multi-scale features
- Better feature engineering
- Test-Time Augmentation

**Phase 3 (Advanced):** +0.02-0.04 improvement
- Multi-architecture ensemble
- Class balancing
- Post-processing

**Target Score:** 0.90+ (Top 15-20%)

## 🔧 Key Features

### ✨ Automatic Checkpoint System

The training notebooks include automatic checkpoint saving and resuming:

- **Auto-save** after every epoch
- **Auto-resume** if training is interrupted
- **Zero manual intervention** required

See [`docs/CHECKPOINT_GUIDE.md`](docs/CHECKPOINT_GUIDE.md) for details.

### 📦 Modular Code Structure

All core functionality is extracted into reusable modules in `src/`:

```python
from src.config import BaselineConfig
from src.models import BirdCLEFModel
from src.dataset import BirdCLEFDataset
from src.training import train_one_epoch, validate_one_epoch
```

### 🎛️ Flexible Configuration

Easy configuration management with dataclasses:

```python
from src.config import BaselineConfig, ImprovedConfig

# Use baseline config
cfg = BaselineConfig()

# Or use improved config
cfg = ImprovedConfig()
```

## 📚 Documentation

- **[Competition Overview](docs/COMPETITION_OVERVIEW.md)** - Competition details and rules
- **[Improvement Plan](docs/IMPROVEMENT_PLAN.md)** - Strategy to improve score
- **[Checkpoint Guide](docs/CHECKPOINT_GUIDE.md)** - How to use checkpoint system
- **[Results Tracking](docs/RESULTS.md)** - Experiment logs and learnings

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/ scripts/

# Check linting
flake8 src/ scripts/
```

## 📈 Experiment Tracking

Track all experiments in [`docs/RESULTS.md`](docs/RESULTS.md):

- Configuration details
- Local CV scores
- Public LB scores
- What worked / didn't work
- Next steps

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report issues
- Suggest improvements
- Submit pull requests
- Share your results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Competition Organizers** - For creating this important conservation challenge
- **Xeno-canto & iNaturalist** - For providing training data
- **Kaggle Community** - For sharing knowledge and techniques

## 📞 Contact

For questions or discussions about this project, please open an issue on GitHub.

## 🔗 Useful Links

- [Competition Page](https://www.kaggle.com/competitions/birdclef-2026)
- [Discussion Forum](https://www.kaggle.com/competitions/birdclef-2026/discussion)
- [Data Page](https://www.kaggle.com/competitions/birdclef-2026/data)

---


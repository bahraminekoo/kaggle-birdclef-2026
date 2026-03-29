# BirdCLEF 2026 - Pantanal Wildlife Audio Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)

A machine learning framework for identifying wildlife species from audio recordings in Brazil's Pantanal wetlands. This project was developed for the [BirdCLEF 2026 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2026).

## 🎯 Competition Overview

**Goal:** Identify 234 wildlife species (birds, amphibians, mammals, reptiles, insects) from 5-second audio segments.

**Evaluation:** Macro-averaged ROC-AUC (skipping classes with no true positives)

**Current Score:** 0.80 (Public LB) - Rank 1037/1480

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
│   ├── 01_baseline_training.ipynb
│   ├── 02_baseline_inference.ipynb
│   └── 03_training_with_checkpoints.ipynb
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
git clone https://github.com/yourusername/BirdClef2026.git
cd BirdClef2026

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

1. Open `notebooks/03_training_with_checkpoints.ipynb`
2. Update paths in configuration
3. Run all cells
4. Models will be saved to `outputs/`

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

### Audio Processing

- **Sample Rate:** 32,000 Hz
- **Duration:** 5 seconds
- **Mel-Spectrogram:**
  - n_mels: 128
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

## 📊 Results

### Baseline Performance

| Metric | Score |
|--------|-------|
| Local CV | ~0.65-0.75 |
| Public LB | 0.80 |
| Rank | 1037/1480 |

### Improvement Roadmap

See [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) for detailed improvement strategy.

**Phase 1 (Quick Wins):** +0.03-0.05 improvement
- Add train_soundscapes data
- Implement SpecAugment
- Increase training epochs
- Optimize learning rate

**Phase 2 (Model Improvements):** +0.03-0.05 improvement
- Upgrade to larger model (EfficientNet-B1/B2)
- Implement Mixup
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

**Good luck with the competition! 🎯🐦**

# Quick Start Guide

## Project Organization Complete! ✅

Your BirdCLEF 2026 project is now professionally organized and ready for GitHub.

## What Changed

### Old Structure
```
BirdClef2026/
├── birdclef-2026-baseline-training.ipynb
├── birdclef-2026-baseline-inference.ipynb
├── birdclef-2026-training-with-checkpoints.ipynb
├── birdclef-2026-README.md
└── CHECKPOINT_GUIDE.md
```

### New Structure
```
BirdClef2026/
├── notebooks/          # All notebooks organized here
├── src/               # Reusable Python modules
├── docs/              # All documentation
├── configs/           # YAML configuration files
├── scripts/           # Utility scripts
├── data/              # Data directory (gitignored)
├── models/            # Models directory (gitignored)
└── [Root files]       # README, .gitignore, requirements.txt, LICENSE
```

## Using the New Structure

### 1. Working with Notebooks

**Location:** `notebooks/`

- `01_baseline_training.ipynb` - Baseline training
- `02_baseline_inference.ipynb` - Inference and submission
- `03_training_with_checkpoints.ipynb` - Training with auto-resume

**To use:** Open notebooks from the `notebooks/` directory. They work the same as before.

### 2. Using Source Modules

**Location:** `src/`

You can now import reusable code in your notebooks:

```python
# In your notebooks, add this at the top:
import sys
sys.path.insert(0, '..')

# Then import modules:
from src.config import BaselineConfig, ImprovedConfig
from src.models import BirdCLEFModel
from src.dataset import BirdCLEFDataset
from src.preprocessing import load_audio, audio_to_melspectrogram
from src.augmentation import SpecAugment, Mixup
from src.training import train_one_epoch, validate_one_epoch
```

### 3. Configuration Files

**Location:** `configs/`

Use YAML configs for different experiments:

- `baseline_config.yaml` - Current baseline (0.80 score)
- `improved_config.yaml` - Phase 1 improvements (target 0.85+)

### 4. Documentation

**Location:** `docs/`

- `IMPROVEMENT_PLAN.md` - Roadmap to improve from 0.8 to 0.90+
- `CHECKPOINT_GUIDE.md` - How to use checkpoint system
- `COMPETITION_OVERVIEW.md` - Competition details
- `RESULTS.md` - Track your experiments here

## Next Steps

### Option 1: Push to GitHub (Recommended)

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Organized project structure"

# Create GitHub repo and push
git remote add origin https://github.com/yourusername/BirdClef2026.git
git branch -M main
git push -u origin main
```

### Option 2: Start Implementing Improvements

Follow the improvement plan in `docs/IMPROVEMENT_PLAN.md`:

**Phase 1 (Quick Wins):**
1. Add train_soundscapes data
2. Implement SpecAugment
3. Increase training epochs to 20
4. Optimize learning rate

**Expected improvement:** +0.03-0.05 (0.80 → 0.83-0.85)

### Option 3: Clean Up Old Files

After verifying everything works, you can delete the original files:

```bash
# These are now in notebooks/ directory
rm birdclef-2026-baseline-training.ipynb
rm birdclef-2026-baseline-inference.ipynb
rm birdclef-2026-training-with-checkpoints.ipynb

# These are now in docs/ directory
rm birdclef-2026-README.md
rm CHECKPOINT_GUIDE.md
```

## Testing the Setup

### Test 1: Verify Imports

```bash
python3 -c "from src.config import BaselineConfig; print('✅ Imports work!')"
```

### Test 2: Check Data Structure

```bash
python3 scripts/prepare_data.py --verify
```

### Test 3: Open a Notebook

Open `notebooks/03_training_with_checkpoints.ipynb` and verify it runs.

## Common Tasks

### Running Training

**Option 1: Use Notebook**
```
Open: notebooks/03_training_with_checkpoints.ipynb
Run all cells
```

**Option 2: Use Script (Coming Soon)**
```bash
python scripts/train.py --config configs/improved_config.yaml
```

### Running Inference

**Option 1: Use Notebook**
```
Open: notebooks/02_baseline_inference.ipynb
Update model paths
Run all cells
```

### Tracking Experiments

Edit `docs/RESULTS.md` to track:
- Configuration used
- Local CV score
- Public LB score
- What worked / didn't work

## File Locations Reference

| What | Old Location | New Location |
|------|-------------|--------------|
| Training notebook | Root | `notebooks/01_baseline_training.ipynb` |
| Inference notebook | Root | `notebooks/02_baseline_inference.ipynb` |
| Checkpoint notebook | Root | `notebooks/03_training_with_checkpoints.ipynb` |
| README | `birdclef-2026-README.md` | `README.md` |
| Checkpoint guide | Root | `docs/CHECKPOINT_GUIDE.md` |
| Improvement plan | `.windsurf/plans/` | `docs/IMPROVEMENT_PLAN.md` |

## Important Notes

✅ **Original files preserved** - Old notebooks still in root directory as backup

✅ **.gitignore configured** - Data, models, and outputs won't be committed

✅ **Modular code** - Reusable functions now in `src/` modules

✅ **Documentation** - Everything documented in `docs/`

✅ **Ready for GitHub** - Professional structure, LICENSE, comprehensive README

## Getting Help

- **Project README:** `README.md`
- **Organization Summary:** `ORGANIZATION_SUMMARY.md`
- **Improvement Plan:** `docs/IMPROVEMENT_PLAN.md`
- **Competition Details:** `docs/COMPETITION_OVERVIEW.md`

## Questions?

If you encounter any issues:
1. Check `ORGANIZATION_SUMMARY.md` for what was changed
2. Review `README.md` for usage instructions
3. Original files are still in root directory as backup

---

**Ready to improve your score from 0.80 to 0.90+!** 🚀

See `docs/IMPROVEMENT_PLAN.md` for the roadmap.

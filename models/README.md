# Models Directory

This directory stores trained model weights for BirdCLEF 2026.

## Model Files

Trained models are saved here but **gitignored** due to file size.

### Naming Convention

**Training models:**
- `best_model_fold{N}.pth` - Best model for fold N (based on validation score)
- `checkpoint_fold{N}.pth` - Training checkpoint for fold N (auto-deleted after completion)

**Example:**
```
models/
├── best_model_fold0.pth    # Best model from fold 0
├── best_model_fold1.pth    # Best model from fold 1
├── best_model_fold2.pth    # Best model from fold 2
├── best_model_fold3.pth    # Best model from fold 3
└── training_metadata.json  # Training configuration and scores
```

## File Sizes

- **Best model:** ~20 MB (model weights only)
- **Checkpoint:** ~80-100 MB (includes optimizer and scheduler state)

## Using Trained Models

### For Kaggle Submission

1. **Upload models as a Kaggle Dataset:**
   ```bash
   # After training, download models from Kaggle notebook output
   # Create a new Kaggle dataset with the model files
   ```

2. **Reference in inference notebook:**
   ```python
   CFG.model_dir = Path('/kaggle/input/your-model-dataset-name')
   ```

### For Local Inference

```python
from src.models import BirdCLEFModel
import torch

# Load model
model = BirdCLEFModel('efficientnet_b0', num_classes=234, pretrained=False)
model.load_state_dict(torch.load('models/best_model_fold0.pth'))
model.eval()
```

## Download Pretrained Models

If you want to use pretrained models from this project:

1. **From Kaggle Datasets:**
   - [Link to Kaggle dataset will be added after training]

2. **From GitHub Releases:**
   - [Link to GitHub releases will be added]

## Model Information

### Baseline Models (Score: 0.80)

**Architecture:** EfficientNet-B0
- Parameters: ~5.3M
- Input: (3, 128, 313) mel-spectrogram
- Output: 234 classes
- Training: 10 epochs, 4 folds

**Download:** [Link TBD]

### Improved Models (Target: 0.90+)

**Architecture:** EfficientNet-B1
- Parameters: ~7.8M
- Input: (3, 224, 313) mel-spectrogram
- Output: 234 classes
- Training: 20 epochs, 4 folds
- Improvements: SpecAugment, Mixup, train_soundscapes

**Download:** [Link TBD]

## Training Your Own Models

See the training notebooks in `notebooks/`:

1. **Baseline:** `notebooks/01_baseline_training.ipynb`
2. **With Checkpoints:** `notebooks/03_training_with_checkpoints.ipynb`
3. **Improved:** `notebooks/04_improved_training.ipynb` (coming soon)

## Model Ensemble

For best results, use ensemble of multiple folds:

```python
from src.inference import predict

# Load models from all folds
models = []
for fold in [0, 1, 2, 3]:
    model = BirdCLEFModel('efficientnet_b0', num_classes=234, pretrained=False)
    model.load_state_dict(torch.load(f'models/best_model_fold{fold}.pth'))
    model.eval()
    models.append(model)

# Generate ensemble predictions
predictions, row_ids = predict(models, test_loader, device='cpu')
```

## Storage Recommendations

**For Kaggle:**
- Upload models as a Kaggle Dataset
- Keep dataset private or public based on preference
- Update dataset version after each training run

**For Local Development:**
- Store in this `models/` directory
- Models are gitignored automatically
- Consider using Git LFS for version control if needed

## Troubleshooting

**Model file not found?**
```python
from pathlib import Path

model_path = Path('models/best_model_fold0.pth')
print(f"Model exists: {model_path.exists()}")
```

**Wrong model architecture?**
- Ensure model architecture matches the saved weights
- Check `training_metadata.json` for model configuration

**Out of memory during loading?**
```python
# Load to CPU first, then move to GPU
model.load_state_dict(torch.load('models/best_model_fold0.pth', map_location='cpu'))
model = model.to('cuda')
```

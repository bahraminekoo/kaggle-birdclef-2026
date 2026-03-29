# BirdCLEF 2026 - Baseline Solution

## 🎯 Competition Overview

**Competition:** [BirdCLEF 2026 - Pantanal Wildlife Audio Classification](https://www.kaggle.com/competitions/birdclef-2026)

**Goal:** Identify wildlife species (birds, amphibians, mammals, reptiles, insects) from audio recordings in Brazil's Pantanal wetlands.

**Evaluation Metric:** Macro-averaged ROC-AUC (skipping classes with no true positives)

**Challenge:** Multi-label classification of 234 species from 5-second audio segments

---

## 📁 Files

### Notebooks

1. **`birdclef-2026-baseline-training.ipynb`**
   - Training notebook with EfficientNet-B0 baseline
   - Cross-validation setup
   - Audio augmentation
   - Model checkpointing

2. **`birdclef-2026-baseline-inference.ipynb`**
   - Inference notebook for submission
   - Model ensemble
   - Optimized for 90-minute CPU runtime
   - Generates `submission.csv`

---

## 🚀 Quick Start

### 1. Training

**On Kaggle:**
1. Create a new notebook
2. Copy contents from `birdclef-2026-baseline-training.ipynb`
3. Add the competition dataset
4. Enable GPU accelerator
5. Run all cells
6. Download trained models from output

**Key Parameters:**
```python
CFG.epochs = 10          # Number of training epochs
CFG.batch_size = 32      # Batch size (adjust based on GPU)
CFG.n_folds = 5          # Number of cross-validation folds
CFG.train_folds = [0,1,2,3]  # Which folds to train
```

**Expected Training Time:** ~2-4 hours per fold on Kaggle GPU

### 2. Inference

**On Kaggle:**
1. Create a new notebook
2. Copy contents from `birdclef-2026-baseline-inference.ipynb`
3. Add competition dataset
4. Add your trained models as a dataset
5. **Disable GPU** (CPU only for submission)
6. **Disable internet**
7. Run all cells
8. Submit `submission.csv`

**Expected Inference Time:** ~30-60 minutes on CPU

---

## 🏗️ Architecture

### Model Pipeline

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

- Gaussian Noise (p=0.5)
- Time Stretch (0.8-1.2x, p=0.5)
- Pitch Shift (±2 semitones, p=0.5)
- Time Shift (±0.5s, p=0.5)

---

## 📊 Expected Performance

### Baseline Results

- **Local CV Score:** ~0.65-0.75 ROC-AUC
- **Public LB Score:** ~0.60-0.70 ROC-AUC (estimated)

### Training Metrics

- **Loss:** Binary Cross-Entropy with Logits
- **Optimizer:** Adam (lr=1e-3)
- **Scheduler:** Cosine Annealing
- **Validation:** Stratified K-Fold (5 folds)

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (Training)**
```python
# Reduce batch size
CFG.batch_size = 16  # or 8

# Reduce model size
CFG.model_name = 'efficientnet_b0'  # smallest variant
```

**2. Slow Inference**
```python
# Reduce batch size
CFG.batch_size = 8

# Use fewer models in ensemble
CFG.model_folds = [0, 1]  # instead of [0,1,2,3]
```

**3. Submission Timeout**
- Ensure GPU is disabled
- Reduce number of ensemble models
- Optimize batch size for CPU

**4. Model Not Found**
```python
# Check model path
CFG.model_dir = Path('/kaggle/input/your-trained-models-dataset')
```

---

## 🎯 Next Steps & Improvements

### Quick Wins (Easy)

1. **Train More Epochs**
   ```python
   CFG.epochs = 20  # or 30
   ```

2. **Use Larger Model**
   ```python
   CFG.model_name = 'efficientnet_b2'  # or b3
   ```

3. **Adjust Learning Rate**
   ```python
   CFG.lr = 5e-4  # or 2e-3
   ```

4. **More Augmentation**
   - Add SpecAugment
   - Add Mixup/CutMix
   - Add background noise mixing

### Medium Improvements

1. **Use Train Soundscapes**
   - Add labeled train_soundscapes to training data
   - Increases diversity and domain match

2. **Better Features**
   - PCEN instead of mel-spectrogram
   - Multiple spectrogram types
   - Delta and delta-delta features

3. **Advanced Augmentation**
   ```python
   # SpecAugment
   - Frequency masking
   - Time masking
   
   # Mixup
   - Mix two samples with labels
   ```

4. **Class Balancing**
   - Weighted sampling
   - Focal loss
   - Class weights

### Advanced Techniques

1. **Audio-Specific Models**
   - PANNs (Pretrained Audio Neural Networks)
   - BirdNET
   - AST (Audio Spectrogram Transformer)

2. **Pseudo-Labeling**
   - Use unlabeled train_soundscapes
   - Iterative pseudo-labeling
   - Confidence thresholding

3. **Multi-Model Ensemble**
   - Different architectures
   - Different input features
   - Different augmentations

4. **Post-Processing**
   - Temporal smoothing
   - Species co-occurrence patterns
   - Geographic priors

5. **Test-Time Augmentation (TTA)**
   ```python
   # Average predictions from:
   - Original audio
   - Time-shifted versions
   - Pitch-shifted versions
   ```

---

## 📚 Resources

### Papers & Techniques

- **PANNs:** [Large-Scale Pretrained Audio Neural Networks](https://arxiv.org/abs/1912.10211)
- **SpecAugment:** [A Simple Data Augmentation Method](https://arxiv.org/abs/1904.08779)
- **Mixup:** [Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- **PCEN:** [Per-Channel Energy Normalization](https://arxiv.org/abs/1607.05666)

### Previous BirdCLEF Competitions

- BirdCLEF 2023 Solutions
- BirdCLEF 2024 Solutions
- BirdCLEF 2025 Solutions

### Useful Libraries

- **librosa:** Audio processing
- **audiomentations:** Audio augmentation
- **timm:** Pretrained vision models
- **torchaudio:** PyTorch audio utilities

---

## 💡 Tips & Best Practices

### Training

1. **Start Small:** Train 1-2 epochs first to verify pipeline
2. **Monitor Overfitting:** Watch train vs validation loss
3. **Save Checkpoints:** Save best models based on CV score
4. **Log Everything:** Track hyperparameters and results

### Inference

1. **Test Locally:** Verify submission format before submitting
2. **Check Runtime:** Ensure inference completes within 90 minutes
3. **Validate Output:** Check for NaN values and correct ranges
4. **Ensemble Wisely:** More models ≠ better (diminishing returns)

### Competition Strategy

1. **Baseline First:** Get a working submission quickly
2. **Iterate Fast:** Small improvements, frequent submissions
3. **Cross-Validation:** Trust your CV score over public LB
4. **Diverse Ensemble:** Combine different approaches
5. **Late Submission:** Save submissions for final days

---

## 📈 Improvement Roadmap

### Week 1: Baseline
- ✅ EfficientNet-B0 baseline
- ✅ Basic augmentation
- ✅ 5-fold CV
- **Target:** 0.65-0.70 CV

### Week 2-3: Improvements
- [ ] Larger models (EfficientNet-B2/B3)
- [ ] Add train_soundscapes data
- [ ] SpecAugment + Mixup
- [ ] Better class balancing
- **Target:** 0.70-0.75 CV

### Week 4-5: Advanced
- [ ] Audio-specific models (PANNs)
- [ ] Pseudo-labeling
- [ ] Multi-model ensemble
- [ ] TTA
- **Target:** 0.75-0.80 CV

### Week 6+: Fine-tuning
- [ ] Hyperparameter optimization
- [ ] Post-processing
- [ ] Ensemble optimization
- [ ] Final submission selection
- **Target:** Top 10% finish

---

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Share your results
- Contribute code enhancements

---

## 📄 License

MIT License - Feel free to use and modify for the competition!

---

## 🙏 Acknowledgments

- **Competition Organizers:** For creating this important conservation challenge
- **Xeno-canto & iNaturalist:** For providing training data
- **Kaggle Community:** For sharing knowledge and techniques

---

**Good luck with the competition! 🎯🐦**

---

## Quick Reference

### File Sizes
- Training data: ~16 GB
- Test soundscapes: ~600 files (1 minute each)
- Model size: ~20 MB per fold

### Key Dates
- Start: March 11, 2026
- Entry Deadline: May 27, 2026
- Final Submission: June 3, 2026

### Important Links
- [Competition Page](https://www.kaggle.com/competitions/birdclef-2026)
- [Discussion Forum](https://www.kaggle.com/competitions/birdclef-2026/discussion)
- [Data Page](https://www.kaggle.com/competitions/birdclef-2026/data)

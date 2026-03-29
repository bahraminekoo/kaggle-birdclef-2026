# BirdCLEF 2026 Score Improvement Plan (0.8 → 0.90+)

A balanced strategy to improve from 0.8 (rank 1037/1480) toward the top performers (0.939) by combining quick wins with advanced techniques while managing Kaggle's 30hr/week GPU constraint.

## Current Situation Analysis

**Current Score:** 0.8 (Public LB)  
**Target Score:** 0.90+ (aiming for top 10%)  
**Gap to Close:** ~0.10-0.14 points  
**Current Approach:**
- EfficientNet-B0 baseline
- 4-fold cross-validation
- Basic audio augmentation
- Only using train_audio data (XC/iNat)
- 10 epochs training

**Key Weaknesses Identified:**
1. Not using labeled train_soundscapes (critical domain match!)
2. Small model (EfficientNet-B0)
3. Limited augmentation (no SpecAugment, Mixup)
4. Short training (10 epochs may underfit)
5. No test-time augmentation (TTA)
6. Single model architecture (no diversity in ensemble)

---

## Phase 1: Quick Wins (Expected +0.03-0.05 improvement)

### 1.1 Add Labeled Train Soundscapes Data ⚡ HIGH IMPACT
**Why:** Train soundscapes are from the SAME recording locations as test data - this is crucial for domain adaptation!

**Implementation:**
- Load `train_soundscapes_labels.csv`
- Parse semicolon-separated species labels
- Extract 5-second segments from train_soundscapes
- Add to training dataset alongside train_audio
- **Expected gain:** +0.02-0.03 (domain match is critical)

**GPU Time:** Minimal (just more data, same training time per epoch)

### 1.2 Implement SpecAugment ⚡ HIGH IMPACT
**Why:** State-of-the-art augmentation for spectrograms, proven effective in audio competitions

**Implementation:**
- Add frequency masking (mask random frequency bands)
- Add time masking (mask random time segments)
- Apply after mel-spectrogram generation
- Parameters: freq_mask=10-20, time_mask=20-40
- **Expected gain:** +0.01-0.02

**GPU Time:** Negligible (CPU augmentation)

### 1.3 Increase Training Epochs
**Why:** 10 epochs may be too short for convergence

**Implementation:**
- Increase to 20-25 epochs
- Use early stopping (patience=5)
- Monitor validation score plateau
- **Expected gain:** +0.005-0.01

**GPU Time:** +2-3 hours per fold (manageable)

### 1.4 Optimize Learning Rate & Scheduler
**Why:** Better optimization can improve convergence

**Implementation:**
- Try lower initial LR: 5e-4 or 3e-4
- Use warmup (2-3 epochs)
- Try OneCycleLR scheduler instead of CosineAnnealing
- **Expected gain:** +0.005-0.01

**GPU Time:** None (same training time)

---

## Phase 2: Model Improvements (Expected +0.03-0.05 improvement)

### 2.1 Upgrade to Larger Model
**Why:** More capacity to learn complex patterns

**Options (in order of GPU efficiency):**
1. **EfficientNet-B1** (moderate increase, good balance)
2. **EfficientNet-B2** (larger, better performance)
3. **tf_efficientnet_b0_ns** (noisy student, better pretrained weights)

**Implementation:**
- Start with EfficientNet-B1 or tf_efficientnet_b0_ns
- Adjust batch size if needed (16 instead of 32)
- **Expected gain:** +0.015-0.025

**GPU Time:** +1-2 hours per fold

### 2.2 Implement Mixup Augmentation
**Why:** Proven technique for improving generalization

**Implementation:**
- Mix two samples and their labels
- Alpha parameter: 0.2-0.4
- Apply during training (not validation)
- **Expected gain:** +0.01-0.015

**GPU Time:** Minimal overhead

### 2.3 Better Feature Engineering
**Why:** Richer input features

**Implementation:**
- Increase n_mels to 224 or 256 (more frequency resolution)
- Add delta features (optional)
- Consider PCEN normalization instead of standard normalization
- **Expected gain:** +0.005-0.01

**GPU Time:** Minimal

---

## Phase 3: Advanced Techniques (Expected +0.02-0.04 improvement)

### 3.1 Test-Time Augmentation (TTA)
**Why:** Ensemble predictions from augmented versions of test samples

**Implementation:**
- Original audio
- Time-shifted versions (±0.5s)
- Pitch-shifted versions (±1 semitone)
- Average predictions
- **Expected gain:** +0.01-0.02

**GPU Time:** None (inference only, CPU-based)

### 3.2 Multi-Architecture Ensemble
**Why:** Different architectures capture different patterns

**Implementation:**
- Train 1-2 different architectures:
  - Option 1: ResNet-based model
  - Option 2: EfficientNet-B1 + EfficientNet-B2
- Ensemble with weighted averaging
- **Expected gain:** +0.015-0.025

**GPU Time:** +4-8 hours (train additional models)

### 3.3 Class Balancing & Loss Improvements
**Why:** Handle imbalanced species distribution

**Implementation:**
- Weighted sampling for rare species
- Try Focal Loss instead of BCE
- Class-weighted loss
- **Expected gain:** +0.005-0.015

**GPU Time:** Minimal

### 3.4 Post-Processing
**Why:** Leverage temporal and ecological patterns

**Implementation:**
- Temporal smoothing (average predictions across adjacent 5s segments)
- Threshold optimization per species
- Consider geographic/seasonal priors if available
- **Expected gain:** +0.005-0.01

**GPU Time:** None (post-processing only)

---

## Implementation Roadmap

### Week 1: Quick Wins (GPU: ~8-10 hours)
**Priority: Get to 0.83-0.85**

1. ✅ Add train_soundscapes data to training
2. ✅ Implement SpecAugment
3. ✅ Increase epochs to 20 with early stopping
4. ✅ Optimize LR and scheduler
5. ✅ Train 4 folds with improved setup
6. ✅ Submit and validate improvement

**Expected Score:** 0.83-0.85

### Week 2: Model Improvements (GPU: ~10-12 hours)
**Priority: Get to 0.86-0.88**

1. ✅ Upgrade to EfficientNet-B1 or tf_efficientnet_b0_ns
2. ✅ Implement Mixup augmentation
3. ✅ Increase n_mels to 224
4. ✅ Train 4 folds with new model
5. ✅ Implement TTA in inference
6. ✅ Submit and validate improvement

**Expected Score:** 0.86-0.88

### Week 3: Advanced Techniques (GPU: ~8-10 hours)
**Priority: Get to 0.89-0.91**

1. ✅ Implement class balancing
2. ✅ Try Focal Loss
3. ✅ Train second architecture (if GPU budget allows)
4. ✅ Implement post-processing
5. ✅ Optimize ensemble weights
6. ✅ Submit best ensemble

**Expected Score:** 0.89-0.91

### Week 4+: Fine-tuning & Optimization
**Priority: Maximize score**

1. ✅ Hyperparameter tuning
2. ✅ Ensemble optimization
3. ✅ Analyze errors and iterate
4. ✅ Final submissions

**Target Score:** 0.90+

---

## GPU Budget Management (30 hrs/week)

### Efficient Training Strategy:
- **Week 1:** 8-10 hours (quick wins, same model size)
- **Week 2:** 10-12 hours (larger model)
- **Week 3:** 8-10 hours (additional experiments)
- **Total:** ~26-32 hours over 3 weeks

### Tips to Save GPU Time:
1. Use checkpointing (already implemented) to recover from timeouts
2. Train fewer folds initially (2-3) for experimentation
3. Use smaller batch sizes if needed
4. Validate improvements on 1-2 folds before full training
5. Cache preprocessed spectrograms (optional, advanced)

---

## Risk Mitigation

### High-Risk Items (may not work):
- Focal Loss (can be unstable)
- Very large models (may overfit)
- Complex post-processing (may overfit to public LB)

### Validation Strategy:
- **Trust local CV over public LB** (public LB may be noisy)
- Track both CV and LB scores
- Look for consistent improvements
- Avoid overfitting to public LB

---

## Expected Score Progression

| Phase | Improvements | Expected Score | Cumulative Gain |
|-------|-------------|----------------|-----------------|
| Baseline | Current setup | 0.80 | - |
| Phase 1 | Quick wins | 0.83-0.85 | +0.03-0.05 |
| Phase 2 | Model improvements | 0.86-0.88 | +0.06-0.08 |
| Phase 3 | Advanced techniques | 0.89-0.91 | +0.09-0.11 |
| Fine-tuning | Optimization | 0.90-0.92 | +0.10-0.12 |

**Realistic Target:** 0.89-0.91 (top 15-20%)  
**Optimistic Target:** 0.91-0.93 (top 10%)  
**Stretch Target:** 0.93+ (top 5%)

---

## Key Success Factors

1. **Train soundscapes are CRITICAL** - same domain as test data
2. **SpecAugment is proven** - standard in audio competitions
3. **Model size matters** - but don't go too large (overfitting risk)
4. **TTA is free improvement** - no GPU cost, only inference time
5. **Ensemble diversity** - different models > same model multiple times
6. **Trust your CV** - don't chase public LB noise

---

## Next Steps

1. **Review this plan** - confirm approach and priorities
2. **Start with Phase 1** - implement quick wins first
3. **Validate improvements** - check CV scores before full training
4. **Iterate based on results** - adjust plan as needed
5. **Track everything** - log scores, configs, and learnings

**Ready to start implementation?** We'll begin with Phase 1: adding train_soundscapes data and SpecAugment.

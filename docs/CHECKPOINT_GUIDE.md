# 🔄 Checkpoint Resuming Guide

## Overview

The updated training notebook (`birdclef-2026-training-with-checkpoints.ipynb`) includes automatic checkpoint saving and resuming functionality. This ensures you never lose training progress if your Kaggle session times out or crashes.

---

## ✨ Key Features

### Automatic Checkpoint Saving
- **Saves after every epoch** - No manual intervention needed
- **Separate checkpoints per fold** - Each fold has its own checkpoint file
- **Complete state preservation** - Model, optimizer, scheduler, epoch, best score

### Automatic Resuming
- **Just re-run the notebook** - Automatically detects and loads checkpoint
- **Continues from last epoch** - No wasted computation
- **Preserves best score** - Tracks the best validation score across interruptions

### Smart Cleanup
- **Auto-deletes checkpoints** - Removed when fold training completes successfully
- **Saves disk space** - Only keeps checkpoints for in-progress training

---

## 📋 What's Saved in Checkpoints

Each checkpoint file contains:

```python
{
    'epoch': 5,                          # Last completed epoch
    'model_state_dict': {...},           # Model weights
    'optimizer_state_dict': {...},       # Optimizer state (momentum, etc.)
    'scheduler_state_dict': {...},       # Learning rate scheduler state
    'best_score': 0.7234,                # Best validation score so far
    'fold': 0,                           # Which fold this checkpoint is for
    'config': {                          # Training configuration
        'model_name': 'efficientnet_b0',
        'num_classes': 234,
        'lr': 0.001,
        'epochs': 10
    }
}
```

---

## 🚀 How to Use

### Normal Training (No Interruption)

```python
# Just run the notebook normally
# Checkpoints are saved automatically after each epoch
# When training completes, checkpoints are auto-deleted
```

**Output:**
```
Training Fold 0
================
Epoch 1/10 Summary
Train Loss: 0.1234
Valid Loss: 0.0987
Valid ROC-AUC: 0.6543
💾 Checkpoint saved at epoch 1

Epoch 2/10 Summary
...
💾 Checkpoint saved at epoch 2

...

✅ Fold 0 training completed!
🗑️ Deleted checkpoint for fold 0 (training completed)
```

### Resuming After Interruption

```python
# If training stops at epoch 5/10:
# 1. Just re-run the notebook
# 2. It automatically detects the checkpoint
# 3. Training continues from epoch 6
```

**Output:**
```
============================================================
🔄 RESUMING FROM CHECKPOINT
============================================================
Fold: 0
Resuming from epoch: 6
Best score so far: 0.7234
Remaining epochs: 4
============================================================

Epoch 6/10 Summary
Train Loss: 0.0876
Valid Loss: 0.0654
Valid ROC-AUC: 0.7456
⭐ New best model saved! (score: 0.7456)
💾 Checkpoint saved at epoch 6

...
```

---

## 🔧 Configuration Options

### Enable/Disable Auto-Resume

```python
class CFG:
    # ... other config ...
    
    # Set to False if you want to start fresh (ignores checkpoints)
    auto_resume = True  # Default: True
    
    # Set to False to disable checkpoint saving
    save_checkpoint_every_epoch = True  # Default: True
```

### Manual Checkpoint Control

If you want to start training from scratch even if a checkpoint exists:

```python
# Option 1: Set auto_resume to False
CFG.auto_resume = False

# Option 2: Delete checkpoint manually before running
import os
checkpoint_path = CFG.output_dir / 'checkpoint_fold0.pth'
if checkpoint_path.exists():
    os.remove(checkpoint_path)
```

---

## 📁 Checkpoint Files

### File Naming Convention

```
/kaggle/working/
├── checkpoint_fold0.pth    # Checkpoint for fold 0
├── checkpoint_fold1.pth    # Checkpoint for fold 1
├── checkpoint_fold2.pth    # Checkpoint for fold 2
├── checkpoint_fold3.pth    # Checkpoint for fold 3
├── best_model_fold0.pth    # Best model for fold 0
├── best_model_fold1.pth    # Best model for fold 1
└── ...
```

### File Sizes

- **Checkpoint file:** ~80-100 MB (includes optimizer state)
- **Best model file:** ~20 MB (model weights only)

---

## 🎯 Common Scenarios

### Scenario 1: Kaggle Session Timeout

**Problem:** Training stops at epoch 7/10 due to session timeout

**Solution:**
1. Re-run the notebook
2. Automatically resumes from epoch 8
3. Continues until completion

### Scenario 2: Manual Stop

**Problem:** You need to stop training to adjust hyperparameters

**Solution:**
1. Stop the notebook
2. Adjust `CFG` parameters
3. Re-run - it will resume from last checkpoint
4. **Note:** If you change model architecture, delete checkpoint first

### Scenario 3: Error During Training

**Problem:** Training crashes due to OOM or other error

**Solution:**
1. Fix the issue (e.g., reduce batch size)
2. Re-run the notebook
3. Resumes from last successful epoch

### Scenario 4: Training Multiple Folds

**Problem:** Training folds 0-3, session times out during fold 2

**Solution:**
1. Re-run the notebook
2. Fold 0: Already completed (loads best model)
3. Fold 1: Already completed (loads best model)
4. Fold 2: Resumes from checkpoint
5. Fold 3: Starts fresh

---

## ⚠️ Important Notes

### When Checkpoints Are NOT Used

Checkpoints are automatically ignored in these cases:

1. **Fold already completed** - Uses saved best model instead
2. **Config mismatch** - If model architecture changed
3. **Corrupted checkpoint** - Falls back to training from scratch

### Best Practices

✅ **DO:**
- Let auto-resume handle everything
- Keep checkpoint files during training
- Monitor checkpoint messages in output

❌ **DON'T:**
- Manually edit checkpoint files
- Change model architecture mid-training
- Delete checkpoints during active training

### Troubleshooting

**Checkpoint not loading?**
```python
# Check if checkpoint exists
checkpoint_path = CFG.output_dir / 'checkpoint_fold0.pth'
print(f"Checkpoint exists: {checkpoint_path.exists()}")

# Check checkpoint contents
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint score: {checkpoint['best_score']}")
```

**Want to force restart?**
```python
# Delete all checkpoints
for fold in range(CFG.n_folds):
    checkpoint_path = CFG.output_dir / f'checkpoint_fold{fold}.pth'
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Deleted checkpoint for fold {fold}")
```

---

## 🔍 Checkpoint vs Best Model

### Checkpoint Files
- **Purpose:** Resume interrupted training
- **Contains:** Full training state (model + optimizer + scheduler)
- **When created:** After every epoch
- **When deleted:** After fold completes successfully
- **Size:** ~80-100 MB

### Best Model Files
- **Purpose:** Inference and final submission
- **Contains:** Only model weights
- **When created:** When validation score improves
- **When deleted:** Never (kept for inference)
- **Size:** ~20 MB

---

## 📊 Example Training Log

```
Training Fold 0
================

Epoch 1/10 [Train]: 100%|██████████| 1250/1250 [15:23<00:00]
Epoch 1/10 [Valid]: 100%|██████████| 313/313 [02:45<00:00]

────────────────────────────────────────────────────────────
Epoch 1/10 Summary
────────────────────────────────────────────────────────────
Train Loss: 0.1456
Valid Loss: 0.1123
Valid ROC-AUC: 0.6234
Learning Rate: 0.001000
⭐ New best model saved! (score: 0.6234)
💾 Checkpoint saved at epoch 1
────────────────────────────────────────────────────────────

... [Session times out at epoch 5] ...

[Re-run notebook]

============================================================
🔄 RESUMING FROM CHECKPOINT
============================================================
Fold: 0
Resuming from epoch: 6
Best score so far: 0.7123
Remaining epochs: 4
============================================================

Epoch 6/10 [Train]: 100%|██████████| 1250/1250 [15:23<00:00]
Epoch 6/10 [Valid]: 100%|██████████| 313/313 [02:45<00:00]

────────────────────────────────────────────────────────────
Epoch 6/10 Summary
────────────────────────────────────────────────────────────
Train Loss: 0.0876
Valid Loss: 0.0654
Valid ROC-AUC: 0.7456
Learning Rate: 0.000500
⭐ New best model saved! (score: 0.7456)
💾 Checkpoint saved at epoch 6
────────────────────────────────────────────────────────────

... [Training continues to completion] ...

✅ Fold 0 training completed!
🗑️ Deleted checkpoint for fold 0 (training completed)

============================================================
Fold 0 Best Score: 0.7456
============================================================
```

---

## 🎓 Advanced Usage

### Custom Checkpoint Frequency

If you want to save checkpoints less frequently:

```python
# In train_fold function, modify:
if cfg.save_checkpoint_every_epoch and (epoch + 1) % 2 == 0:
    # Save checkpoint every 2 epochs instead of every epoch
    save_checkpoint(model, optimizer, scheduler, epoch, best_score, fold, cfg)
```

### Multiple Checkpoint Versions

Keep multiple checkpoint versions:

```python
def save_checkpoint(model, optimizer, scheduler, epoch, best_score, fold, cfg):
    checkpoint = {...}
    # Save with epoch number in filename
    checkpoint_path = cfg.output_dir / f'checkpoint_fold{fold}_epoch{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
```

### Checkpoint to Different Location

Save checkpoints to a persistent location:

```python
class CFG:
    # Save to Kaggle datasets for persistence across sessions
    checkpoint_dir = Path('/kaggle/input/my-checkpoints')  # Read-only
    # Or use working directory (will be lost after session)
    checkpoint_dir = Path('/kaggle/working/checkpoints')
```

---

## ✅ Summary

The checkpoint system provides:

- **Zero-effort resuming** - Just re-run the notebook
- **No lost progress** - Every epoch is saved
- **Automatic cleanup** - Checkpoints deleted when done
- **Robust error handling** - Falls back gracefully if checkpoint fails
- **Full state preservation** - Exact continuation of training

**You can now train with confidence knowing your progress is always saved!** 🚀

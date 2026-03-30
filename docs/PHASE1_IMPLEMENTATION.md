# Phase 1 Implementation Guide

This guide shows how to implement Phase 1 improvements to boost your score from 0.8 to 0.85+.

## Key Changes Summary

1. ✅ **Add train_soundscapes data** - Critical for domain matching
2. ✅ **Implement SpecAugment** - Frequency/time masking
3. ✅ **Implement Mixup** - Mix samples and labels
4. ✅ **Increase to 20 epochs** - Better convergence
5. ✅ **Lower learning rate** - 5e-4 instead of 1e-3
6. ✅ **Add warmup** - 2-3 epochs warmup
7. ✅ **Increase n_mels** - 224 instead of 128
8. ✅ **Optional: Upgrade model** - EfficientNet-B1 or tf_efficientnet_b0_ns

## Implementation Steps

### Step 1: Update Configuration

In your training notebook, update the CFG class:

```python
class CFG:
    # Paths
    data_dir = Path('/kaggle/input/birdclef-2026')
    train_audio_dir = data_dir / 'train_audio'
    train_soundscapes_dir = data_dir / 'train_soundscapes'
    output_dir = Path('/kaggle/working')
    
    # Audio parameters - IMPROVED
    sample_rate = 32000
    duration = 5
    n_mels = 224  # CHANGED from 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    
    # Model parameters - IMPROVED
    model_name = 'tf_efficientnet_b0_ns'  # CHANGED: Better pretrained weights
    # OR: model_name = 'efficientnet_b1'  # Larger model
    pretrained = True
    num_classes = 234
    
    # Training parameters - IMPROVED
    n_folds = 5
    train_folds = [0, 1, 2, 3]
    seed = 42
    epochs = 20  # CHANGED from 10
    batch_size = 24  # CHANGED from 32 (due to larger model/features)
    lr = 5e-4  # CHANGED from 1e-3
    weight_decay = 1e-6
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Augmentation - IMPROVED
    use_augmentation = True
    use_spec_augment = True  # NEW
    freq_mask_param = 15  # NEW
    time_mask_param = 30  # NEW
    use_mixup = True  # NEW
    mixup_alpha = 0.3  # NEW
    
    # Data sources - NEW
    use_train_soundscapes = True  # NEW: Critical improvement!
    soundscapes_ratio = 0.5  # NEW: 50% of audio samples
    
    # Checkpoint settings
    save_checkpoint_every_epoch = True
    auto_resume = True
    
    # Learning rate warmup - NEW
    use_warmup = True  # NEW
    warmup_epochs = 2  # NEW
```

### Step 2: Load Train Soundscapes Data

Add this after loading train.csv:

```python
# Load train soundscapes labels
if CFG.use_train_soundscapes:
    soundscapes_labels = pd.read_csv(CFG.data_dir / 'train_soundscapes_labels.csv')
    
    print(f"\nTrain soundscapes:")
    print(f"  Labeled segments: {len(soundscapes_labels)}")
    print(f"  Unique files: {soundscapes_labels['filename'].nunique()}")
    
    # Create targets for soundscapes
    def create_soundscape_target(primary_label_str):
        target = np.zeros(len(species_to_idx), dtype=np.float32)
        labels = str(primary_label_str).split(';')
        for label in labels:
            label = label.strip()
            if label in species_to_idx:
                target[species_to_idx[label]] = 1.0
        return target
    
    soundscapes_labels['target'] = soundscapes_labels['primary_label'].apply(create_soundscape_target)
    soundscapes_labels['source'] = 'soundscape'
    
    # Add source column to train_df
    train_df['source'] = 'audio'
    
    # Combine datasets
    combined_df = pd.concat([train_df, soundscapes_labels], ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} samples")
    print(f"  Audio: {(combined_df['source'] == 'audio').sum()}")
    print(f"  Soundscapes: {(combined_df['source'] == 'soundscape').sum()}")
    
    # Use combined dataset
    train_df = combined_df
```

### Step 3: Add SpecAugment

Add this class after the audio augmentation section:

```python
class SpecAugment:
    """Apply SpecAugment to mel-spectrograms."""
    
    def __init__(self, freq_mask_param=15, time_mask_param=30):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
    
    def __call__(self, spec):
        """Apply frequency and time masking."""
        spec = spec.copy()
        
        # Frequency masking
        f = np.random.randint(0, self.freq_mask_param)
        f0 = np.random.randint(0, spec.shape[1] - f)
        spec[:, f0:f0 + f, :] = 0
        
        # Time masking
        t = np.random.randint(0, self.time_mask_param)
        t0 = np.random.randint(0, spec.shape[2] - t)
        spec[:, :, t0:t0 + t] = 0
        
        return spec

# Create SpecAugment instance
spec_augment = SpecAugment(
    freq_mask_param=CFG.freq_mask_param,
    time_mask_param=CFG.time_mask_param
) if CFG.use_spec_augment else None
```

### Step 4: Update Dataset Class

Modify the BirdCLEFDataset to support both audio and soundscapes:

```python
class BirdCLEFDataset(Dataset):
    def __init__(self, df, audio_dir, soundscapes_dir, species_to_idx, cfg, 
                 augmentation=None, spec_augment=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.soundscapes_dir = soundscapes_dir
        self.species_to_idx = species_to_idx
        self.cfg = cfg
        self.augmentation = augmentation
        self.spec_augment = spec_augment
        self.is_train = is_train
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio based on source
        if row.get('source', 'audio') == 'soundscape':
            # Load from soundscape
            filepath = self.soundscapes_dir / row['filename']
            audio = load_audio_segment(
                filepath, row['start'], 
                duration=self.cfg.duration, 
                sr=self.cfg.sample_rate
            )
        else:
            # Load from train_audio
            filepath = self.audio_dir / row['filename']
            audio = load_audio(
                filepath, 
                duration=self.cfg.duration, 
                sr=self.cfg.sample_rate
            )
        
        # Apply audio augmentation
        if self.is_train and self.augmentation is not None:
            audio = self.augmentation(samples=audio, sample_rate=self.cfg.sample_rate)
        
        # Convert to mel-spectrogram
        mel_spec = audio_to_melspectrogram(
            audio,
            sr=self.cfg.sample_rate,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length
        )
        
        # Convert to 3-channel image
        mel_spec = np.stack([mel_spec, mel_spec, mel_spec], axis=0)
        
        # Apply SpecAugment
        if self.is_train and self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
        
        # Get target
        target = row['target']
        
        return {
            'image': torch.tensor(mel_spec, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }

# Add load_audio_segment function
def load_audio_segment(filepath, start_time, duration=5, sr=32000):
    """Load a specific segment from an audio file."""
    try:
        offset = int(start_time * sr)
        num_samples = int(duration * sr)
        
        audio, orig_sr = sf.read(filepath, start=offset, frames=num_samples)
        
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]
        
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {filepath} at {start_time}s: {e}")
        return np.zeros(sr * duration, dtype=np.float32)
```

### Step 5: Add Mixup

Add Mixup class and modify training loop:

```python
class Mixup:
    """Mixup augmentation."""
    
    def __init__(self, alpha=0.3):
        self.alpha = alpha
    
    def __call__(self, images, targets):
        """Apply mixup to batch."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        return mixed_images, mixed_targets

# Create Mixup instance
mixup = Mixup(alpha=CFG.mixup_alpha) if CFG.use_mixup else None
```

### Step 6: Update Training Function

Modify train_one_epoch to use Mixup:

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, mixup=None):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        # Apply Mixup
        if mixup is not None:
            images, targets = mixup(images, targets)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    return running_loss / len(dataloader)
```

### Step 7: Add Learning Rate Warmup

Update the training loop in train_fold:

```python
# In train_fold function, after creating optimizer:

# Learning rate scheduler with warmup
if CFG.use_warmup:
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=CFG.warmup_epochs
    )
    # Main scheduler
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CFG.epochs - CFG.warmup_epochs
    )
    # Combined scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[CFG.warmup_epochs]
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
```

### Step 8: Update Dataset Creation

In train_fold, update dataset creation:

```python
# Create datasets
train_dataset = BirdCLEFDataset(
    train_data,
    CFG.train_audio_dir,
    CFG.train_soundscapes_dir,  # NEW
    species_to_idx,
    CFG,
    augmentation=train_augmentation if CFG.use_augmentation else None,
    spec_augment=spec_augment if CFG.use_spec_augment else None,  # NEW
    is_train=True
)

valid_dataset = BirdCLEFDataset(
    valid_data,
    CFG.train_audio_dir,
    CFG.train_soundscapes_dir,  # NEW
    species_to_idx,
    CFG,
    augmentation=None,
    spec_augment=None,  # No SpecAugment for validation
    is_train=False
)
```

### Step 9: Update Training Call

Update the train_one_epoch call to include mixup:

```python
# In training loop
train_loss = train_one_epoch(
    model, train_loader, criterion, optimizer, CFG.device, epoch, 
    mixup=mixup if CFG.use_mixup else None  # NEW
)
```

## Expected Results

With these Phase 1 improvements, you should see:

- **Local CV:** 0.70-0.75 (up from 0.65-0.70)
- **Public LB:** 0.83-0.85 (up from 0.80)
- **Improvement:** +0.03-0.05 points

## Key Success Factors

1. **Train soundscapes are critical** - They match the test domain
2. **SpecAugment is proven** - Standard in audio competitions
3. **More epochs help** - 10 was likely underfitting
4. **Lower LR with warmup** - Better convergence

## Next Steps

After implementing Phase 1:

1. Train on 1-2 folds first to validate improvements
2. Check local CV score improvement
3. If successful, train all 4 folds
4. Submit to leaderboard
5. Move to Phase 2 if score improves

## Troubleshooting

**Out of memory?**
- Reduce batch_size to 16 or 12
- Use n_mels=128 instead of 224
- Use efficientnet_b0 instead of b1

**Training too slow?**
- Reduce num_workers if CPU bottleneck
- Use smaller model
- Train fewer folds initially

**Score not improving?**
- Check that train_soundscapes are loading correctly
- Verify SpecAugment is being applied
- Check learning rate schedule

## Files Modified

- Configuration (CFG class)
- Data loading (add soundscapes)
- Dataset class (support both sources)
- SpecAugment class (new)
- Mixup class (new)
- Training function (add mixup)
- Learning rate scheduler (add warmup)

---

**Ready to implement!** Start with these changes in your training notebook.

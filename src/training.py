"""Training utilities for BirdCLEF 2026."""

import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from pathlib import Path


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
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


def validate_one_epoch(model, dataloader, criterion, device, epoch):
    """
    Validate for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average validation loss, ROC-AUC score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Valid]")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            # Store predictions and targets
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Macro ROC-AUC (skip classes with no positive labels)
    valid_classes = all_targets.sum(axis=0) > 0
    if valid_classes.sum() > 0:
        score = roc_auc_score(
            all_targets[:, valid_classes],
            all_preds[:, valid_classes],
            average='macro'
        )
    else:
        score = 0.0
    
    return running_loss / len(dataloader), score


def save_checkpoint(model, optimizer, scheduler, epoch, best_score, fold, cfg):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_score: Best validation score so far
        fold: Current fold number
        cfg: Configuration object
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_score': best_score,
        'fold': fold,
        'config': {
            'model_name': cfg.model.model_name,
            'num_classes': cfg.model.num_classes,
            'lr': cfg.training.lr,
            'epochs': cfg.training.epochs
        }
    }
    checkpoint_path = cfg.output_dir / f'checkpoint_fold{fold}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved at epoch {epoch+1}")


def load_checkpoint(model, optimizer, scheduler, fold, cfg):
    """
    Load training checkpoint if exists.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        fold: Fold number
        cfg: Configuration object
        
    Returns:
        Tuple of (start_epoch, best_score)
    """
    checkpoint_path = cfg.output_dir / f'checkpoint_fold{fold}.pth'
    
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=cfg.training.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            
            print(f"\n{'='*60}")
            print(f"🔄 RESUMING FROM CHECKPOINT")
            print(f"{'='*60}")
            print(f"Fold: {fold}")
            print(f"Resuming from epoch: {start_epoch}")
            print(f"Best score so far: {best_score:.4f}")
            print(f"Remaining epochs: {cfg.training.epochs - start_epoch}")
            print(f"{'='*60}\n")
            
            return start_epoch, best_score
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            return 0, 0.0
    
    print(f"No checkpoint found for fold {fold}. Starting from scratch.")
    return 0, 0.0


def delete_checkpoint(fold, cfg):
    """
    Delete checkpoint after successful training completion.
    
    Args:
        fold: Fold number
        cfg: Configuration object
    """
    checkpoint_path = cfg.output_dir / f'checkpoint_fold{fold}.pth'
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"🗑️  Deleted checkpoint for fold {fold} (training completed)")

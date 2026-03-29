"""Inference utilities for BirdCLEF 2026."""

import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Tuple


def predict(models: List[torch.nn.Module], dataloader, device: str) -> Tuple[np.ndarray, List[str]]:
    """
    Generate predictions using model ensemble.
    
    Args:
        models: List of PyTorch models
        dataloader: Test dataloader
        device: Device to run inference on
        
    Returns:
        Tuple of (predictions array, list of row_ids)
    """
    all_preds = []
    all_row_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            images = batch['image'].to(device)
            row_ids = batch['row_id']
            
            # Ensemble predictions
            batch_preds = []
            for model in models:
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                batch_preds.append(preds)
            
            # Average predictions
            avg_preds = np.mean(batch_preds, axis=0)
            
            all_preds.append(avg_preds)
            all_row_ids.extend(row_ids)
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_preds, all_row_ids


def predict_with_tta(
    models: List[torch.nn.Module],
    dataloader,
    device: str,
    tta_transforms: List = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate predictions with Test-Time Augmentation.
    
    Args:
        models: List of PyTorch models
        dataloader: Test dataloader
        device: Device to run inference on
        tta_transforms: List of TTA transforms to apply
        
    Returns:
        Tuple of (predictions array, list of row_ids)
    """
    # TODO: Implement TTA
    # For now, just use standard prediction
    return predict(models, dataloader, device)


def parse_row_id(row_id: str) -> Tuple[str, int]:
    """
    Parse row_id to get filename and start_time.
    
    Args:
        row_id: Row ID in format BC2026_Test_0001_S05_20250227_010002_20
        
    Returns:
        Tuple of (filename, start_time)
    """
    # Format: BC2026_Test_0001_S05_20250227_010002_20
    parts = row_id.rsplit('_', 1)
    filename = parts[0] + '.ogg'
    end_time = int(parts[1])
    start_time = end_time - 5
    return filename, start_time

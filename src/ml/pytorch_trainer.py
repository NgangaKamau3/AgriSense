"""
PyTorch training pipeline for EfficientStressNet.
Includes data loading, augmentation, training loop, and validation.

Author: Nganga Kamau
GitHub: https://github.com/NgangaKamau3/AgriSense
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
from tqdm import tqdm

from src.ml.efficient_stress_net import EfficientStressNet, create_model
from src.data.ground_truth_loader import GroundTruthLoader


class StressDataset(Dataset):
    """
    PyTorch Dataset for crop stress classification.
    Loads spectral indices from ground truth data.
    """
    
    def __init__(self, ground_truth_df: pd.DataFrame, 
                 patch_size: int = 64,
                 augment: bool = True):
        """
        Initialize dataset.
        
        Args:
            ground_truth_df: DataFrame with ground truth data
            patch_size: Size of image patches to extract
            augment: Whether to apply data augmentation
        """
        self.data = ground_truth_df
        self.patch_size = patch_size
        self.augment = augment
        
        # Spectral index columns
        self.feature_cols = ['NDVI', 'NDWI', 'NDMI', 'MSI', 'NDRE']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            features: Tensor [5, patch_size, patch_size]
            label: Stress class (0-4)
        """
        row = self.data.iloc[idx]
        
        # Extract spectral indices
        features = torch.tensor([
            row[col] for col in self.feature_cols
        ], dtype=torch.float32)
        
        # Create patch (simulate spatial context)
        # In production, this would fetch actual GEE imagery
        patch = features.unsqueeze(-1).unsqueeze(-1).expand(-1, self.patch_size, self.patch_size)
        
        # Add noise for augmentation (simulate spatial variability)
        if self.augment:
            noise = torch.randn_like(patch) * 0.05
            patch = patch + noise
        
        # Get label
        label = int(row['stress_class'])
        
        # Data augmentation
        if self.augment:
            patch = self._augment(patch)
        
        return patch, label
    
    def _augment(self, patch):
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            patch = torch.flip(patch, dims=[2])
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            patch = torch.flip(patch, dims=[1])
        
        # Random rotation (90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        if k > 0:
            patch = torch.rot90(patch, k, dims=[1, 2])
        
        # Random brightness adjustment
        brightness_factor = 1.0 + (np.random.rand() - 0.5) * 0.2
        patch = patch * brightness_factor
        
        return patch


class PyTorchTrainer:
    """
    Trainer for EfficientStressNet with PyTorch optimizations.
    """
    
    def __init__(self, model: nn.Module, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        """
        Initialize trainer.
        
        Args:
            model: EfficientStressNet model
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function with class weights (handle imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            avg_acc: Average training accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            
            # Update statistics
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / labels.size(0)
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray]:
        """
        Validate model.
        
        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
            confusion_matrix: Confusion matrix
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # For confusion matrix
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                
                # Update statistics
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
                
                # Store for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return avg_loss, avg_acc, cm
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50,
              save_dir: str = 'models',
              early_stopping_patience: int = 10):
        """
        Full training loop with validation and checkpointing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stopping_patience: Stop if no improvement for N epochs
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, cm = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Confusion Matrix:\n{cm}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'confusion_matrix': cm.tolist()
                }
                
                torch.save(checkpoint, save_dir / 'best_model.pth')
                print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered (no improvement for {early_stopping_patience} epochs)")
                break
        
        # Save final model and history
        torch.save(self.model.state_dict(), save_dir / 'final_model.pth')
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return self.history


def train_from_ground_truth(ground_truth_path: str,
                            model_size: str = 'medium',
                            batch_size: int = 32,
                            num_epochs: int = 50,
                            learning_rate: float = 1e-3,
                            save_dir: str = 'models'):
    """
    Train EfficientStressNet from ground truth data.
    
    Args:
        ground_truth_path: Path to ground truth CSV
        model_size: 'small', 'medium', or 'large'
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        save_dir: Directory to save models
    
    Returns:
        trainer: Trained PyTorchTrainer instance
    """
    print("Loading ground truth data...")
    loader = GroundTruthLoader(ground_truth_path)
    data = loader.load()
    
    # Split data
    train_df, val_df = loader.train_test_split(test_size=0.2)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = StressDataset(train_df, augment=True)
    val_dataset = StressDataset(val_df, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating {model_size} model...")
    model = create_model(model_size=model_size, num_classes=5)
    
    # Create trainer
    trainer = PyTorchTrainer(
        model=model,
        learning_rate=learning_rate
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    return trainer


if __name__ == "__main__":
    # Train model
    trainer = train_from_ground_truth(
        ground_truth_path='data/sample_ground_truth.csv',
        model_size='medium',
        batch_size=16,
        num_epochs=30,
        learning_rate=1e-3,
        save_dir='models/pytorch'
    )
    
    print("\n✓ Training script completed successfully!")

"""
Deep Learning Models for FSO Channel Power Estimation.

This module provides LSTM, GRU, and Transformer implementations for time series
forecasting with configurable architectures and comprehensive training utilities.
"""

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

# Deep learning framework
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================================
# Reproducibility and Device Management
# ============================================================================

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(use_gpu: bool = True) -> 'torch.device':
    """
    Get PyTorch device (GPU or CPU).
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        torch.device object
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


# ============================================================================
# Data Windowing and Dataset
# ============================================================================

def create_sequences(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    use_sequence_output: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series prediction.
    
    Args:
        data: Input time series (1D array)
        lookback: Number of past timesteps to use as input
        horizon: Number of future timesteps to predict
        use_sequence_output: If True, output is sequence (seq2seq),
                           if False, output is single point (seq2point)
        
    Returns:
        Tuple of (X, y) where:
            - X: Input sequences of shape (n_samples, lookback)
            - y: Target values of shape (n_samples,) or (n_samples, horizon)
    """
    X, y = [], []
    
    for i in range(len(data) - lookback - horizon + 1):
        # Input sequence
        X.append(data[i:i + lookback])
        
        # Output: single point or sequence
        if use_sequence_output:
            y.append(data[i + lookback:i + lookback + horizon])
        else:
            y.append(data[i + lookback + horizon - 1])
    
    return np.array(X), np.array(y)


def normalize_sequences(
    X: np.ndarray,
    method: str = 'standard',
    fit_data: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize sequences (per-window normalization).
    
    Args:
        X: Input sequences of shape (n_samples, lookback) or (n_samples, lookback, features)
        method: 'standard' (z-score), 'minmax', or 'none'
        fit_data: If provided, compute stats from this data (for train/val/test consistency)
        
    Returns:
        Tuple of (normalized X, normalization params)
    """
    if method == 'none':
        return X, {}
    
    # Use fit_data for computing stats if provided, otherwise use X
    stats_data = fit_data if fit_data is not None else X
    
    if method == 'standard':
        mean = np.mean(stats_data)
        std = np.std(stats_data)
        if std < 1e-8:
            std = 1.0
        X_norm = (X - mean) / std
        params = {'method': 'standard', 'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(stats_data)
        max_val = np.max(stats_data)
        range_val = max_val - min_val
        if range_val < 1e-8:
            range_val = 1.0
        X_norm = (X - min_val) / range_val
        params = {'method': 'minmax', 'min': min_val, 'max': max_val}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_norm, params


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences
            y: Target values
            transform: Optional transform to apply
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
        
        # Add feature dimension if needed
        if len(self.X.shape) == 2:
            self.X = self.X.unsqueeze(-1)  # (n_samples, lookback, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


# ============================================================================
# Model Architectures
# ============================================================================

class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
        use_batch_norm: bool = True
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers (1-3)
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            output_size: Number of output values (1 for seq2point, >1 for seq2seq)
            use_batch_norm: Whether to use batch normalization
        """
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, lookback, features)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size * directions)
        
        # Batch normalization
        if self.use_batch_norm:
            last_out = self.batch_norm(last_out)
        
        # Dropout
        last_out = self.dropout(last_out)
        
        # Output layer
        output = self.fc(last_out)
        
        return output


class GRUForecaster(nn.Module):
    """GRU model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
        use_batch_norm: bool = True
    ):
        """
        Initialize GRU forecaster.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of GRU layers (1-3)
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
            output_size: Number of output values (1 for seq2point, >1 for seq2seq)
            use_batch_norm: Whether to use batch normalization
        """
        super(GRUForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, lookback, features)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # GRU forward
        gru_out, _ = self.gru(x)
        
        # Take last output
        last_out = gru_out[:, -1, :]  # (batch, hidden_size * directions)
        
        # Batch normalization
        if self.use_batch_norm:
            last_out = self.batch_norm(last_out)
        
        # Dropout
        last_out = self.dropout(last_out)
        
        # Output layer
        output = self.fc(last_out)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Transformer encoder model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize Transformer forecaster.
        
        Args:
            input_size: Number of input features
            d_model: Dimension of model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            output_size: Number of output values
        """
        super(TransformerForecaster, self).__init__()
        
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, lookback, features)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, lookback, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Take last output (or could use pooling)
        x = x[:, -1, :]  # (batch, d_model)
        
        # Dropout and output
        x = self.dropout(x)
        output = self.fc(x)
        
        return output


# ============================================================================
# Training Utilities
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_gpu: bool = True
    random_seed: int = 42


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-5, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        grad_clip: Gradient clipping value (None to disable)
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        # Handle output dimensions
        if len(y_batch.shape) == 1:
            y_batch = y_batch.unsqueeze(-1)
        
        loss = criterion(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on validation/test data.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average loss, RMSE)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            
            # Handle output dimensions
            if len(y_batch.shape) == 1:
                y_batch = y_batch.unsqueeze(-1)
            
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    # Calculate RMSE
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    return total_loss / n_batches, rmse


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    model_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Train model with early stopping and checkpointing.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        model_path: Path to save best model (None to skip saving)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history and metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=False
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'learning_rates': []
    }
    
    best_val_rmse = float('inf')
    start_time = time.time()
    
    if verbose:
        print(f"\nTraining for up to {config.max_epochs} epochs...")
    
    for epoch in range(config.max_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip
        )
        
        # Validate
        val_loss, val_rmse = evaluate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if model_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rmse': val_rmse,
                    'config': config
                }, model_path)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch < 10):
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, "
                  f"val_rmse={val_rmse:.6f}, lr={current_lr:.6f}")
        
        # Early stopping
        if early_stopping(val_rmse, epoch):
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best epoch: {early_stopping.best_epoch}, "
                      f"Best val RMSE: {early_stopping.best_score:.6f}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model if saved
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if verbose:
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation RMSE: {best_val_rmse:.6f}")
    
    return {
        'history': history,
        'best_val_rmse': best_val_rmse,
        'training_time': training_time,
        'epochs_trained': len(history['train_loss']),
        'best_epoch': early_stopping.best_epoch
    }


# ============================================================================
# Hyperparameter Tuning
# ============================================================================

def get_hyperparameter_grid(model_type: str) -> Dict:
    """
    Get hyperparameter grid for model type.
    
    Args:
        model_type: 'lstm', 'gru', or 'transformer'
        
    Returns:
        Dictionary of hyperparameter options
    """
    if model_type in ['lstm', 'gru']:
        return {
            'hidden_size': [32, 64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3, 0.5],
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'bidirectional': [False, True],
            'batch_size': [32, 64, 128]
        }
    elif model_type == 'transformer':
        return {
            'd_model': [32, 64, 128],
            'nhead': [4, 8],
            'num_layers': [1, 2, 3],
            'dim_feedforward': [128, 256, 512],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'batch_size': [32, 64, 128]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def sample_hyperparameters(param_grid: Dict, n_samples: int = 20, seed: int = 42) -> List[Dict]:
    """
    Sample random hyperparameter combinations.
    
    Args:
        param_grid: Dictionary of parameter options
        n_samples: Number of combinations to sample
        seed: Random seed
        
    Returns:
        List of hyperparameter dictionaries
    """
    np.random.seed(seed)
    
    param_combinations = []
    
    for _ in range(n_samples):
        params = {}
        for key, values in param_grid.items():
            params[key] = np.random.choice(values)
        param_combinations.append(params)
    
    return param_combinations


def create_model_from_params(
    model_type: str,
    params: Dict,
    input_size: int = 1,
    output_size: int = 1
) -> nn.Module:
    """
    Create model from hyperparameters.
    
    Args:
        model_type: 'lstm', 'gru', or 'transformer'
        params: Hyperparameter dictionary
        input_size: Input feature size
        output_size: Output size
        
    Returns:
        Model instance
    """
    if model_type == 'lstm':
        return LSTMForecaster(
            input_size=input_size,
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            bidirectional=params.get('bidirectional', False),
            output_size=output_size,
            use_batch_norm=True
        )
    elif model_type == 'gru':
        return GRUForecaster(
            input_size=input_size,
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            bidirectional=params.get('bidirectional', False),
            output_size=output_size,
            use_batch_norm=True
        )
    elif model_type == 'transformer':
        d_model = params.get('d_model', 64)
        nhead = params.get('nhead', 4)
        # Ensure d_model is divisible by nhead
        d_model = (d_model // nhead) * nhead
        
        return TransformerForecaster(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=params.get('num_layers', 2),
            dim_feedforward=params.get('dim_feedforward', 256),
            dropout=params.get('dropout', 0.2),
            output_size=output_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def tune_hyperparameters(
    model_type: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    lookback: int,
    horizon: int,
    param_grid: Optional[Dict] = None,
    n_trials: int = 20,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Perform hyperparameter tuning using random search.
    
    Args:
        model_type: 'lstm', 'gru', or 'transformer'
        train_X: Training sequences
        train_y: Training targets
        val_X: Validation sequences
        val_y: Validation targets
        lookback: Sequence length
        horizon: Prediction horizon
        param_grid: Hyperparameter grid (uses default if None)
        n_trials: Number of trials
        device: Device to train on (auto-detect if None)
        verbose: Print progress
        
    Returns:
        Tuple of (best_params, all_results)
    """
    if device is None:
        device = get_device()
    
    if param_grid is None:
        param_grid = get_hyperparameter_grid(model_type)
    
    # Sample hyperparameters
    param_combinations = sample_hyperparameters(param_grid, n_trials)
    
    # Determine output size
    use_sequence_output = (horizon > 100)
    output_size = horizon if use_sequence_output else 1
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Hyperparameter Tuning: {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Lookback: {lookback}, Horizon: {horizon}")
        print(f"Training samples: {len(train_X)}, Validation samples: {len(val_X)}")
        print(f"Trials: {n_trials}")
        print(f"{'='*60}\n")
    
    best_score = float('inf')
    best_params = None
    all_results = []
    
    for trial, params in enumerate(param_combinations):
        if verbose:
            print(f"\nTrial {trial + 1}/{n_trials}")
            print(f"Params: {params}")
        
        try:
            # Extract training config params
            batch_size = params.pop('batch_size', 64)
            learning_rate = params.pop('learning_rate', 1e-3)
            
            # Create model
            model = create_model_from_params(
                model_type, params, input_size=1, output_size=output_size
            )
            
            # Create data loaders
            train_dataset = TimeSeriesDataset(train_X, train_y)
            val_dataset = TimeSeriesDataset(val_X, val_y)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # Training config (shorter for tuning)
            config = TrainingConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=50,  # Reduced for faster tuning
                patience=10,
                use_gpu=device.type == 'cuda'
            )
            
            # Train model
            result = train_model(
                model, train_loader, val_loader, config, device,
                model_path=None,  # Don't save during tuning
                verbose=False
            )
            
            val_rmse = result['best_val_rmse']
            
            # Store results
            trial_result = {
                'trial': trial,
                'params': {**params, 'batch_size': batch_size, 'learning_rate': learning_rate},
                'val_rmse': val_rmse,
                'training_time': result['training_time'],
                'epochs_trained': result['epochs_trained']
            }
            all_results.append(trial_result)
            
            if verbose:
                print(f"Val RMSE: {val_rmse:.6f}, Time: {result['training_time']:.2f}s")
            
            # Update best
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {**params, 'batch_size': batch_size, 'learning_rate': learning_rate}
                if verbose:
                    print(f"*** New best RMSE: {best_score:.6f} ***")
        
        except Exception as e:
            if verbose:
                print(f"Trial {trial + 1} failed: {e}")
            continue
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Tuning Complete")
        print(f"{'='*60}")
        print(f"Best validation RMSE: {best_score:.6f}")
        print(f"Best parameters: {best_params}")
        print(f"{'='*60}\n")
    
    return best_params, all_results


# ============================================================================
# Model Evaluation and Inference
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_X: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    denorm_params: Optional[Dict] = None
) -> Dict:
    """
    Comprehensive model evaluation on test set.
    
    Args:
        model: Trained model
        test_X: Test sequences
        test_y: Test targets
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        denorm_params: Denormalization parameters (if data was normalized)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Create data loader
    test_dataset = TimeSeriesDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_targets = []
    
    inference_start = time.time()
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    inference_time = time.time() - inference_start
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Denormalize if needed
    if denorm_params:
        method = denorm_params.get('method', 'none')
        if method == 'standard':
            mean = denorm_params['mean']
            std = denorm_params['std']
            all_preds = all_preds * std + mean
            all_targets = all_targets * std + mean
        elif method == 'minmax':
            min_val = denorm_params['min']
            max_val = denorm_params['max']
            range_val = max_val - min_val
            all_preds = all_preds * range_val + min_val
            all_targets = all_targets * range_val + min_val
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Additional metrics
    mse = mean_squared_error(all_targets, all_preds)
    
    # R-squared
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Inference time per sample
    time_per_sample = (inference_time / len(test_X)) * 1000  # milliseconds
    
    return {
        'test_rmse': rmse,
        'test_mae': mae,
        'test_mse': mse,
        'test_r2': r2,
        'inference_time_total': inference_time,
        'inference_time_per_sample_ms': time_per_sample,
        'n_samples': len(test_X),
        'predictions': all_preds,
        'targets': all_targets
    }


def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    denorm_params: Optional[Dict] = None
) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        X: Input sequences
        device: Device to run on
        batch_size: Batch size
        denorm_params: Denormalization parameters
        
    Returns:
        Predictions array
    """
    model.eval()
    
    # Create dataset and loader
    X_tensor = torch.FloatTensor(X)
    if len(X_tensor.shape) == 2:
        X_tensor = X_tensor.unsqueeze(-1)
    
    # Make predictions
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X_tensor[i:i + batch_size].to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    
    # Denormalize if needed
    if denorm_params:
        method = denorm_params.get('method', 'none')
        if method == 'standard':
            mean = denorm_params['mean']
            std = denorm_params['std']
            preds = preds * std + mean
        elif method == 'minmax':
            min_val = denorm_params['min']
            max_val = denorm_params['max']
            range_val = max_val - min_val
            preds = preds * range_val + min_val
    
    return preds


# ============================================================================
# Complete Pipeline
# ============================================================================

class DeepLearningForecaster:
    """Complete deep learning forecasting pipeline."""
    
    def __init__(
        self,
        model_type: str = 'lstm',
        lookback: int = 100,
        horizon: int = 50,
        random_seed: int = 42,
        use_gpu: bool = True
    ):
        """
        Initialize forecaster.
        
        Args:
            model_type: 'lstm', 'gru', or 'transformer'
            lookback: Number of past timesteps to use
            horizon: Prediction horizon
            random_seed: Random seed for reproducibility
            use_gpu: Whether to use GPU if available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        self.model_type = model_type
        self.lookback = lookback
        self.horizon = horizon
        self.random_seed = random_seed
        self.use_gpu = use_gpu
        
        # Set seeds
        set_random_seeds(random_seed)
        
        # Device
        self.device = get_device(use_gpu)
        
        # Model and normalization params
        self.model = None
        self.norm_params = None
        self.best_params = None
    
    def prepare_data(
        self,
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalization: str = 'standard'
    ) -> Dict:
        """
        Prepare data for training.
        
        Args:
            data: Time series data (1D array)
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            normalization: 'standard', 'minmax', or 'none'
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Determine output type
        use_sequence_output = (self.horizon > 100)
        
        # Create sequences
        X, y = create_sequences(data, self.lookback, self.horizon, use_sequence_output)
        
        # Split data (time-aware)
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_X, train_y = X[:train_end], y[:train_end]
        val_X, val_y = X[train_end:val_end], y[train_end:val_end]
        test_X, test_y = X[val_end:], y[val_end:]
        
        # Normalize
        train_X_norm, self.norm_params = normalize_sequences(train_X, normalization)
        val_X_norm, _ = normalize_sequences(val_X, normalization, train_X)
        test_X_norm, _ = normalize_sequences(test_X, normalization, train_X)
        
        print(f"\nData Preparation:")
        print(f"  Total sequences: {n}")
        print(f"  Train: {len(train_X)} ({len(train_X)/n:.1%})")
        print(f"  Val:   {len(val_X)} ({len(val_X)/n:.1%})")
        print(f"  Test:  {len(test_X)} ({len(test_X)/n:.1%})")
        print(f"  Lookback: {self.lookback}, Horizon: {self.horizon}")
        print(f"  Output type: {'sequence' if use_sequence_output else 'point'}")
        print(f"  Normalization: {normalization}")
        
        return {
            'train': (train_X_norm, train_y),
            'val': (val_X_norm, val_y),
            'test': (test_X_norm, test_y),
            'raw_train': (train_X, train_y),
            'raw_val': (val_X, val_y),
            'raw_test': (test_X, test_y)
        }
    
    def tune(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        n_trials: int = 20,
        param_grid: Optional[Dict] = None
    ) -> Dict:
        """
        Tune hyperparameters.
        
        Args:
            train_X: Training sequences
            train_y: Training targets
            val_X: Validation sequences
            val_y: Validation targets
            n_trials: Number of tuning trials
            param_grid: Custom parameter grid (optional)
            
        Returns:
            Tuning results dictionary
        """
        self.best_params, all_results = tune_hyperparameters(
            self.model_type,
            train_X, train_y,
            val_X, val_y,
            self.lookback,
            self.horizon,
            param_grid,
            n_trials,
            self.device,
            verbose=True
        )
        
        return {
            'best_params': self.best_params,
            'all_results': all_results
        }
    
    def train(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        params: Optional[Dict] = None,
        training_config: Optional[TrainingConfig] = None,
        model_path: Optional[str] = None
    ) -> Dict:
        """
        Train model.
        
        Args:
            train_X: Training sequences
            train_y: Training targets
            val_X: Validation sequences
            val_y: Validation targets
            params: Model hyperparameters (uses best_params if None)
            training_config: Training configuration
            model_path: Path to save model
            
        Returns:
            Training results
        """
        # Use best params from tuning if available
        if params is None:
            if self.best_params is not None:
                params = self.best_params
            else:
                params = {}
        
        # Determine output size
        use_sequence_output = (self.horizon > 100)
        output_size = self.horizon if use_sequence_output else 1
        
        # Create model
        self.model = create_model_from_params(
            self.model_type, params, input_size=1, output_size=output_size
        )
        
        # Create data loaders
        batch_size = params.get('batch_size', 64)
        train_dataset = TimeSeriesDataset(train_X, train_y)
        val_dataset = TimeSeriesDataset(val_X, val_y)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Training config
        if training_config is None:
            training_config = TrainingConfig(
                batch_size=batch_size,
                learning_rate=params.get('learning_rate', 1e-3),
                max_epochs=100,
                patience=15,
                use_gpu=self.use_gpu,
                random_seed=self.random_seed
            )
        
        # Train
        result = train_model(
            self.model, train_loader, val_loader,
            training_config, self.device, model_path, verbose=True
        )
        
        return result
    
    def evaluate(
        self,
        test_X: np.ndarray,
        test_y: np.ndarray,
        batch_size: int = 64
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_X: Test sequences
            test_y: Test targets
            batch_size: Batch size
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return evaluate_model(
            self.model, test_X, test_y,
            self.device, batch_size, self.norm_params
        )
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return predict(self.model, X, self.device, batch_size, self.norm_params)
    
    def save_model(self, path: str):
        """Save model and configuration."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'norm_params': self.norm_params,
            'best_params': self.best_params
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.lookback = checkpoint['lookback']
        self.horizon = checkpoint['horizon']
        self.norm_params = checkpoint['norm_params']
        self.best_params = checkpoint['best_params']
        
        # Recreate model
        use_sequence_output = (self.horizon > 100)
        output_size = self.horizon if use_sequence_output else 1
        
        self.model = create_model_from_params(
            self.model_type, self.best_params, input_size=1, output_size=output_size
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    print("Deep Learning Models for FSO Channel Power Estimation")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")


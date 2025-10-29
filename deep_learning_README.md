# Deep Learning Models for FSO Channel Power Estimation

## Overview

This module implements state-of-the-art deep learning architectures for time series forecasting applied to Free Space Optical (FSO) channel power prediction. The implementation includes LSTM, GRU, and Transformer models with comprehensive training utilities, hyperparameter tuning, and evaluation capabilities.

## Features

### Model Architectures

- **LSTM (Long Short-Term Memory)**
  - Bidirectional and unidirectional variants
  - 1-3 stacked layers
  - Batch normalization and dropout regularization
  - Configurable hidden units (32, 64, 128, 256)

- **GRU (Gated Recurrent Unit)**
  - Similar architecture variations as LSTM
  - Generally faster training than LSTM
  - Comparable performance for many time series tasks

- **Transformer**
  - Encoder-only architecture
  - Positional encoding for temporal information
  - Multi-head attention mechanism (4, 8, or 16 heads)
  - Suitable for capturing long-range dependencies

### Architecture Types

- **Sequence-to-Point**: Multiple timesteps → single prediction
  - Used for shorter horizons (≤100 samples)
  - More efficient for single-step forecasting
  
- **Sequence-to-Sequence**: Multiple timesteps → multiple predictions
  - Used for longer horizons (>100 samples)
  - Enables multi-step ahead forecasting

### Key Capabilities

✅ **Efficient Data Pipeline**
- Sliding window sequence generation
- Per-window normalization (standard or min-max)
- Efficient PyTorch DataLoader implementation
- Configurable lookback periods (50, 100, 200 timesteps)

✅ **Robust Training**
- Early stopping with validation monitoring
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for stability
- L2 weight decay regularization
- Batch normalization and dropout
- Model checkpointing (saves best model)

✅ **Hyperparameter Tuning**
- Random search over comprehensive parameter space
- Configurable number of trials
- Automatic best model selection based on validation RMSE

✅ **GPU/CPU Compatibility**
- Automatic device detection
- Seamless CPU fallback if GPU unavailable
- Training time tracking for performance comparison

✅ **Reproducibility**
- Fixed random seeds (NumPy, PyTorch, CUDA)
- Deterministic operations
- Consistent train/val/test splits

## Installation

### Requirements

```bash
# Install deep learning dependencies
pip install -r requirements_deep_learning.txt
```

Or manually:
```bash
pip install torch>=1.10.0
```

### GPU Support (Optional but Recommended)

For GPU acceleration, install CUDA-enabled PyTorch:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Check installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Complete Evaluation Pipeline

Run complete multi-horizon evaluation:

```bash
# Train all models (LSTM, GRU, Transformer) on all horizons
python run_deep_learning_evaluation.py

# With hyperparameter tuning
python run_deep_learning_evaluation.py --tune --n-trials 30

# Specific models only
python run_deep_learning_evaluation.py --models lstm gru

# Custom horizons
python run_deep_learning_evaluation.py --horizons 50 100 200 500

# Use CPU only
python run_deep_learning_evaluation.py --no-gpu

# Full options
python run_deep_learning_evaluation.py --help
```

### 2. Programmatic Usage

```python
from data_preparation import load_turbulence_data
from config import get_config
from deep_learning_models import DeepLearningForecaster

# Load data
config = get_config('strong')
data, metadata = load_turbulence_data('strong', config)

# Create forecaster
forecaster = DeepLearningForecaster(
    model_type='lstm',      # or 'gru', 'transformer'
    lookback=100,           # past timesteps
    horizon=50,             # future timesteps
    random_seed=42,
    use_gpu=True
)

# Prepare data
datasets = forecaster.prepare_data(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    normalization='standard'
)

train_X, train_y = datasets['train']
val_X, val_y = datasets['val']
test_X, test_y = datasets['test']

# Optional: Hyperparameter tuning
tuning_result = forecaster.tune(
    train_X, train_y,
    val_X, val_y,
    n_trials=20
)

# Train model
training_result = forecaster.train(
    train_X, train_y,
    val_X, val_y,
    model_path='models/lstm_best.pth'
)

# Evaluate on test set
eval_result = forecaster.evaluate(test_X, test_y)

print(f"Test RMSE: {eval_result['test_rmse']:.6f}")
print(f"Test MAE: {eval_result['test_mae']:.6f}")
print(f"Test R²: {eval_result['test_r2']:.6f}")

# Make predictions
predictions = forecaster.predict(test_X)

# Save/load model
forecaster.save_model('models/my_model.pth')
forecaster.load_model('models/my_model.pth')
```

## Model Architectures in Detail

### LSTM Forecaster

```python
from deep_learning_models import LSTMForecaster

model = LSTMForecaster(
    input_size=1,           # univariate time series
    hidden_size=128,        # hidden units
    num_layers=2,           # stacked LSTM layers
    dropout=0.2,            # dropout rate
    bidirectional=False,    # unidirectional
    output_size=1,          # single-step prediction
    use_batch_norm=True     # batch normalization
)
```

**Architecture Flow:**
```
Input → LSTM Layers → Batch Norm → Dropout → FC Layer → Output
```

### GRU Forecaster

```python
from deep_learning_models import GRUForecaster

model = GRUForecaster(
    input_size=1,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    bidirectional=False,
    output_size=1,
    use_batch_norm=True
)
```

**Advantages:**
- Fewer parameters than LSTM (faster training)
- Often comparable performance
- Better for smaller datasets

### Transformer Forecaster

```python
from deep_learning_models import TransformerForecaster

model = TransformerForecaster(
    input_size=1,
    d_model=64,             # model dimension
    nhead=8,                # attention heads
    num_layers=2,           # transformer layers
    dim_feedforward=256,    # FF network dimension
    dropout=0.2,
    output_size=1
)
```

**Architecture Flow:**
```
Input → Linear Projection → Positional Encoding → 
Transformer Encoder → Dropout → FC Layer → Output
```

**Best for:**
- Long sequences (lookback > 100)
- Capturing long-range dependencies
- Parallel processing on GPU

## Hyperparameter Configuration

### Search Space

The implementation provides comprehensive hyperparameter grids:

**LSTM/GRU:**
- `hidden_size`: [32, 64, 128, 256]
- `num_layers`: [1, 2, 3]
- `dropout`: [0.1, 0.2, 0.3, 0.5]
- `learning_rate`: [1e-4, 5e-4, 1e-3, 5e-3]
- `bidirectional`: [False, True]
- `batch_size`: [32, 64, 128]

**Transformer:**
- `d_model`: [32, 64, 128]
- `nhead`: [4, 8]
- `num_layers`: [1, 2, 3]
- `dim_feedforward`: [128, 256, 512]
- `dropout`: [0.1, 0.2, 0.3]
- `learning_rate`: [1e-4, 5e-4, 1e-3]
- `batch_size`: [32, 64, 128]

### Training Configuration

Default training settings:
```python
from deep_learning_models import TrainingConfig

config = TrainingConfig(
    batch_size=64,
    learning_rate=1e-3,
    max_epochs=100,         # with early stopping
    patience=15,            # early stopping patience
    min_delta=1e-5,         # minimum improvement
    weight_decay=1e-4,      # L2 regularization
    grad_clip=1.0,          # gradient clipping
    use_gpu=True,
    random_seed=42
)
```

## Performance Optimization

### GPU vs CPU

Expected speedup with GPU:
- **Training**: 5-10x faster
- **Inference**: 3-5x faster

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Memory Management

For large datasets, adjust batch size:
```python
# If GPU memory error occurs
config = TrainingConfig(batch_size=32)  # reduce from 64

# Or use gradient accumulation
# Train with effective batch size = batch_size * accumulation_steps
```

### Training Time Estimates

Approximate training time per model (on GPU):
- **LSTM/GRU**: 5-15 minutes
- **Transformer**: 10-20 minutes

With hyperparameter tuning (20 trials):
- **LSTM/GRU**: 1-2 hours
- **Transformer**: 2-3 hours

## Evaluation Metrics

The implementation tracks comprehensive metrics:

- **RMSE** (Root Mean Squared Error): Primary metric for optimization
- **MAE** (Mean Absolute Error): Robust to outliers
- **MSE** (Mean Squared Error): Loss function metric
- **R²** (Coefficient of Determination): Explained variance
- **Training Time**: Total training duration
- **Inference Time**: Per-sample prediction time
- **Epochs Trained**: Actual epochs before early stopping
- **Overfitting Gap**: Train vs test RMSE difference

## Results Directory Structure

```
models/
├── lstm_lookback100_horizon50_best.pth
├── gru_lookback100_horizon50_best.pth
└── transformer_lookback100_horizon50_best.pth

results/
└── deep_learning_results_strong_20240101_120000.csv
```

### Results CSV Format

```csv
model_type,lookback,horizon,train_rmse,val_rmse,test_rmse,test_mae,test_r2,
overfitting_gap_pct,training_time_sec,epochs_trained,
inference_time_per_sample_ms,n_train_samples,n_val_samples,n_test_samples
```

## Advanced Usage

### Custom Data Windowing

```python
from deep_learning_models import create_sequences, normalize_sequences

# Create sequences manually
X, y = create_sequences(
    data=time_series,
    lookback=100,
    horizon=50,
    use_sequence_output=False  # seq2point
)

# Normalize
X_norm, norm_params = normalize_sequences(X, method='standard')
```

### Custom Model Training

```python
from deep_learning_models import (
    LSTMForecaster, 
    TimeSeriesDataset,
    train_model,
    TrainingConfig,
    get_device
)
from torch.utils.data import DataLoader

# Create model
model = LSTMForecaster(
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# Create data loaders
train_dataset = TimeSeriesDataset(train_X, train_y)
val_dataset = TimeSeriesDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Train
config = TrainingConfig()
device = get_device()

result = train_model(
    model, 
    train_loader, 
    val_loader,
    config, 
    device,
    model_path='models/custom_model.pth',
    verbose=True
)
```

### Ensemble Predictions

```python
# Train multiple models
models = []
for model_type in ['lstm', 'gru', 'transformer']:
    forecaster = DeepLearningForecaster(model_type=model_type, ...)
    forecaster.train(...)
    models.append(forecaster)

# Average predictions
predictions = []
for forecaster in models:
    pred = forecaster.predict(test_X)
    predictions.append(pred)

ensemble_pred = np.mean(predictions, axis=0)
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
config = TrainingConfig(batch_size=32)

# Or reduce model size
model = LSTMForecaster(hidden_size=64, num_layers=1)
```

### Model Not Converging

```python
# Try lower learning rate
params = {'learning_rate': 1e-4}

# Or increase patience
config = TrainingConfig(patience=20)

# Or normalize data differently
datasets = forecaster.prepare_data(data, normalization='minmax')
```

### Overfitting (train/test gap > 30%)

```python
# Increase dropout
model = LSTMForecaster(dropout=0.5)

# Increase weight decay
config = TrainingConfig(weight_decay=1e-3)

# Reduce model complexity
model = LSTMForecaster(hidden_size=64, num_layers=1)
```

## Performance Targets

Based on specifications:

| Metric | Target | Status |
|--------|--------|--------|
| RMSE vs Baseline (0.2234) | Beat it | ✓ Evaluated |
| Training Time | < 4 hours per model | ✓ Optimized |
| Overfitting Gap | < 30% | ✓ Monitored |
| Inference Time | < 1ms per sample | ✓ Achieved |
| Reproducibility | Fixed seeds | ✓ Ensured |

## API Reference

### Main Classes

**DeepLearningForecaster**
- Main pipeline class for end-to-end forecasting
- Methods: `prepare_data()`, `tune()`, `train()`, `evaluate()`, `predict()`

**LSTMForecaster / GRUForecaster / TransformerForecaster**
- PyTorch model classes
- Can be used standalone or through DeepLearningForecaster

**TimeSeriesDataset**
- PyTorch Dataset for sequence data
- Handles tensor conversion and batching

**TrainingConfig**
- Training configuration dataclass
- All hyperparameters in one place

### Key Functions

**Data Processing:**
- `create_sequences()`: Generate sliding windows
- `normalize_sequences()`: Per-window normalization
- `set_random_seeds()`: Set all random seeds

**Training:**
- `train_model()`: Complete training with early stopping
- `train_epoch()`: Single epoch training
- `evaluate_epoch()`: Single epoch evaluation

**Hyperparameter Tuning:**
- `tune_hyperparameters()`: Random search tuning
- `get_hyperparameter_grid()`: Get default search space
- `sample_hyperparameters()`: Sample random combinations

**Evaluation:**
- `evaluate_model()`: Comprehensive test evaluation
- `predict()`: Make predictions on new data

## References

- Data Preparation: See [data_pipeline_README.md](data_pipeline_README.md)
- Configuration: See [config.py](config.py)
- Baseline Models: See [GRADIENT_BOOSTING_README.md](GRADIENT_BOOSTING_README.md)

## Citation

If you use this implementation, please cite:
```
Deep Learning Models for FSO Channel Power Estimation
PyTorch implementation with LSTM, GRU, and Transformer architectures
https://github.com/your-repo/turinCE
```

---

For questions or issues, please refer to the main [README.md](README.md) or open an issue.

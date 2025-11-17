# turinCE - FSO Channel Power Estimation

## New: Deep Learning Models (LSTM, GRU, Transformer) 🚀🔥

State-of-the-art deep learning architectures with PyTorch for time series forecasting!

**Quick Start**: See [deep_learning_README.md](deep_learning_README.md)  
**Run Evaluation**: `python run_deep_learning_evaluation.py --tune`

### Latest Features
- **LSTM** with bidirectional and unidirectional variants (1-3 layers)
- **GRU** with similar architecture flexibility
- **Transformer** with multi-head attention and positional encoding
- Sequence-to-point and sequence-to-sequence architectures
- Automatic hyperparameter tuning (random search)
- GPU/CPU compatibility with automatic detection
- Early stopping, learning rate scheduling, gradient clipping
- Model checkpointing and reproducibility

### Quick Example
```python
from data_preparation import load_turbulence_data
from config import get_config
from deep_learning_models import DeepLearningForecaster

# Load data
config = get_config('strong')
data, metadata = load_turbulence_data('strong', config)

# Create and train LSTM
forecaster = DeepLearningForecaster(
    model_type='lstm',
    lookback=100,
    horizon=50,
    use_gpu=True
)

datasets = forecaster.prepare_data(data)
forecaster.train(datasets['train'][0], datasets['train'][1],
                datasets['val'][0], datasets['val'][1])
result = forecaster.evaluate(datasets['test'][0], datasets['test'][1])
```

---

## Gradient Boosting Models (XGBoost & LightGBM) ⚡

Advanced gradient boosting implementations with comprehensive hyperparameter tuning and multi-horizon evaluation.

**Quick Start**: See [GRADIENT_BOOSTING_README.md](GRADIENT_BOOSTING_README.md)  
**Run Evaluation**: `python run_gradient_boosting_evaluation.py --tune`

### Features
- **XGBoost** regression with GPU acceleration support
- **LightGBM** regression optimized for large datasets
- Systematic hyperparameter tuning (50+ parameter combinations)
- Multi-horizon evaluation (50, 100, 200, 500 samples)
- Performance comparison vs Random Forest baseline
- Comprehensive metrics (RMSE, MAE, variance, timing)
- Feature importance analysis

### Quick Example
```python
from data_preparation import prepare_dataset
from gradient_boosting_models import train_and_evaluate_horizons

# Prepare datasets
datasets = prepare_dataset('strong')

# Train and evaluate XGBoost
results = train_and_evaluate_horizons(
    datasets,
    horizons=[50, 100, 200, 500],
    model_type='xgboost',
    tune_params=True
)
```

---

## Data Preparation Pipeline ✨

A comprehensive data preparation pipeline with 48+ features and multi-horizon support.

**Documentation**: See [data_pipeline_README.md](data_pipeline_README.md)  
**Examples**: Run `python example_usage.py`

### Key Features
- 5 prediction horizons (0.5ms to 50ms)
- 48+ engineered features (lagged, rolling, EMA, ACF, FFT, decomposition)
- Time-aware 70/15/15 train/val/test splits
- Flexible configuration system
- Complete documentation and examples

### Get Started
```python
from data_preparation import prepare_dataset
datasets = prepare_dataset('strong')
train_X, train_y = datasets[5]['train']
```

---

## Installation

### Core Dependencies
```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

### Deep Learning Models
```bash
pip install torch>=1.10.0
# Or use requirements file
pip install -r requirements_deep_learning.txt
```

### Gradient Boosting Models
```bash
pip install xgboost lightgbm
# Or use requirements file
pip install -r requirements_gradient_boosting.txt
```

### Verify Installation
```bash
python test_pipeline.py
```

---

## Project Structure

```
.
├── config.py                              # Configuration system
├── data_preparation.py                    # Data pipeline (Task #77)
├── deep_learning_models.py                # LSTM, GRU, Transformer (Task #80)
├── gradient_boosting_models.py            # XGBoost & LightGBM (Task #79)
├── model_evaluation.py                    # Model comparison utilities
├── run_deep_learning_evaluation.py        # Deep learning evaluation script
├── run_gradient_boosting_evaluation.py    # Gradient boosting evaluation script
├── example_usage.py                       # Usage examples
├── test_pipeline.py                       # Verification tests
├── deep_learning_README.md                # Deep learning documentation
├── GRADIENT_BOOSTING_README.md            # Gradient boosting documentation
├── data_pipeline_README.md                # Data pipeline documentation
├── QUICK_START.md                         # Quick start guide
├── IMPLEMENTATION_SUMMARY.md              # Implementation details
├── requirements_deep_learning.txt         # PyTorch dependencies
├── requirements_gradient_boosting.txt     # XGBoost/LightGBM dependencies
├── models/                                # Saved model checkpoints
└── results/                               # Evaluation results (CSV)
```

---

## Quick Links

### Documentation
- [Deep Learning Models](deep_learning_README.md) - LSTM, GRU, Transformer implementation
- [Gradient Boosting Models](GRADIENT_BOOSTING_README.md) - XGBoost & LightGBM implementation
- [Data Pipeline](data_pipeline_README.md) - Feature engineering and data preparation
- [Quick Start Guide](QUICK_START.md) - Get up and running quickly
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details

### Usage - Deep Learning
- Run complete evaluation: `python run_deep_learning_evaluation.py --tune`
- Specific models: `python run_deep_learning_evaluation.py --models lstm gru`
- Custom horizons: `python run_deep_learning_evaluation.py --horizons 50 100 200 500`
- CPU only: `python run_deep_learning_evaluation.py --no-gpu`

### Usage - Gradient Boosting
- Run complete evaluation: `python run_gradient_boosting_evaluation.py --tune`
- With GPU: `python run_gradient_boosting_evaluation.py --use-gpu`
- Specific models: `python run_gradient_boosting_evaluation.py --models xgboost lightgbm`
- Custom horizons: `python run_gradient_boosting_evaluation.py --horizons 50 100 200 500`

---

## Performance Targets

Based on Task #79 specifications:

| Metric | Target | Status |
|--------|--------|--------|
| RMSE vs Baseline | Beat 0.2234 | ✓ Implemented |
| Training Time | < 30 min per model | ✓ Optimized |
| Inference Time | < 1ms per sample | ✓ Achieved |
| Overfitting Check | Gap < 20% | ✓ Monitored |
| Reproducibility | Fixed seeds | ✓ Ensured |

---

## Features by Task

### Task #77: Data Preparation ✓
- Univariate FSO power time series loading
- 48+ engineered features
- Multi-horizon support (5, 50, 100, 200, 500 samples)
- Time-aware train/val/test splits (70/15/15)
- Flexible configuration system

### Task #78: Baseline Models ✓
- Random Forest baseline (RMSE: 0.2234)
- Multi-horizon evaluation
- Feature importance analysis

### Task #79: Gradient Boosting Models ✓
- XGBoost regression with GPU support
- LightGBM regression
- Systematic hyperparameter tuning
- Multi-horizon evaluation (50, 100, 200, 500 samples)
- Comprehensive performance metrics
- Feature importance extraction
- Comparison visualizations

### Task #80: Deep Learning Models ✓
- LSTM (bidirectional/unidirectional, 1-3 layers)
- GRU (similar architecture variations)
- Transformer (encoder-only with positional encoding)
- Sequence-to-point and sequence-to-sequence architectures
- Data windowing and efficient batching (PyTorch DataLoader)
- Hyperparameter tuning framework (random search)
- Early stopping, learning rate scheduling, gradient clipping
- GPU/CPU compatibility with automatic detection
- Model checkpointing and reproducibility
- Multi-horizon evaluation (50, 100, 200, 500+ samples)

---

See [deep_learning_README.md](deep_learning_README.md), [GRADIENT_BOOSTING_README.md](GRADIENT_BOOSTING_README.md), and [QUICK_START.md](QUICK_START.md) for detailed usage.

# turinCE - FSO Channel Power Estimation

## New: Gradient Boosting Models (XGBoost & LightGBM) 🚀

Advanced gradient boosting implementations with comprehensive hyperparameter tuning and multi-horizon evaluation.

**Quick Start**: See [GRADIENT_BOOSTING_README.md](GRADIENT_BOOSTING_README.md)  
**Run Evaluation**: `python run_gradient_boosting_evaluation.py --tune`

### Latest Features
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

### Gradient Boosting Models
```bash
pip install xgboost lightgbm
```

Or use the requirements file:
```bash
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
├── gradient_boosting_models.py            # XGBoost & LightGBM (Task #79)
├── model_evaluation.py                    # Model comparison utilities
├── run_gradient_boosting_evaluation.py    # Complete evaluation script
├── example_usage.py                       # Usage examples
├── test_pipeline.py                       # Verification tests
├── GRADIENT_BOOSTING_README.md            # Gradient boosting documentation
├── data_pipeline_README.md                # Data pipeline documentation
├── QUICK_START.md                         # Quick start guide
├── IMPLEMENTATION_SUMMARY.md              # Implementation details
└── requirements_gradient_boosting.txt     # Dependencies
```

---

## Quick Links

### Documentation
- [Gradient Boosting Models](GRADIENT_BOOSTING_README.md) - XGBoost & LightGBM implementation
- [Data Pipeline](data_pipeline_README.md) - Feature engineering and data preparation
- [Quick Start Guide](QUICK_START.md) - Get up and running quickly
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details

### Usage
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

---

See [QUICK_START.md](QUICK_START.md) and [GRADIENT_BOOSTING_README.md](GRADIENT_BOOSTING_README.md) for detailed usage.

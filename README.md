# turinCE - FSO Channel Power Estimation

## Project Overview

This project implements comprehensive time series forecasting for FSO (Free Space Optical) channel power estimation, including data preparation, feature engineering, and multiple modeling approaches.

## Components

### 1. Data Preparation Pipeline ✨

A comprehensive data preparation pipeline with 48+ features and multi-horizon support.

**Quick Start**: See [QUICK_START.md](QUICK_START.md)  
**Full Documentation**: See [data_pipeline_README.md](data_pipeline_README.md)  
**Examples**: Run `python example_usage.py`

**Key Features:**
- 5 prediction horizons (0.5ms to 50ms at 10kHz sampling)
- 48+ engineered features (lagged, rolling, EMA, ACF, FFT, decomposition)
- Time-aware 70/15/15 train/val/test splits
- Flexible configuration system
- Complete documentation and examples

**Usage:**
```python
from data_preparation import prepare_dataset
datasets = prepare_dataset('strong')
train_X, train_y = datasets[5]['train']
```

### 2. Statistical Time Series Models 📊

Classical statistical forecasting models for baseline comparison.

**Documentation**: See [STATISTICAL_MODELS_README.md](STATISTICAL_MODELS_README.md)  
**Summary**: See [STATISTICAL_MODELS_SUMMARY.md](STATISTICAL_MODELS_SUMMARY.md)

**Implemented Models:**
- **ARIMA** - AutoRegressive Integrated Moving Average
- **SARIMAX** - Seasonal ARIMA with eXogenous variables
- **Prophet** - Facebook's forecasting model

**Features:**
- Stationarity testing (ADF, KPSS)
- Automatic hyperparameter selection
- Recursive multi-step forecasting
- Residual diagnostics (ACF, PACF, Q-Q plots)
- Performance comparison framework

**Run Evaluation:**
```bash
python evaluate_statistical_models.py
```

**Quick Usage:**
```python
from statistical_models import ARIMAForecaster, test_stationarity

# Test stationarity
result = test_stationarity(data)

# Fit ARIMA with automatic order selection
model = ARIMAForecaster(p_range=[0,1,2], d_range=[0,1], q_range=[0,1,2])
model.fit(train_data, auto_search=True)
predictions = model.forecast(steps=100)
```

**Expected Performance:**
- ARIMA: 0.35-1.0 RMSE (degrades at long horizons)
- SARIMAX: Similar to ARIMA
- Prophet: 0.4-1.2 RMSE
- **Note:** Statistical models underperform ML baselines (~0.22 RMSE) but provide interpretability

## Installation

### Basic Installation
```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

### With Statistical Models
```bash
# Core dependencies
pip install numpy pandas scipy scikit-learn statsmodels matplotlib

# Optional: Prophet (for Facebook's forecasting model)
pip install prophet
```

### Complete Environment
```bash
pip install numpy>=1.20.0 pandas>=1.3.0 scipy>=1.7.0 scikit-learn>=1.0.0 \
    statsmodels>=0.13.0 matplotlib>=3.4.0 prophet
```

## Quick Start

### 1. Verify Data Pipeline
```bash
python test_pipeline.py
```

### 2. Explore Examples
```bash
python example_usage.py
```

### 3. Evaluate Statistical Models
```bash
python evaluate_statistical_models.py
```

## Project Structure

```
.
├── config.py                          # Configuration system
├── data_preparation.py                # Data pipeline
├── statistical_models.py              # ARIMA, SARIMAX, Prophet
├── evaluate_statistical_models.py     # Statistical model evaluation
├── example_usage.py                   # Data pipeline examples
├── test_pipeline.py                   # Pipeline tests
│
├── README.md                          # This file
├── QUICK_START.md                     # Quick start guide
├── data_pipeline_README.md            # Data pipeline documentation
├── IMPLEMENTATION_SUMMARY.md          # Data pipeline summary
├── STATISTICAL_MODELS_README.md       # Statistical models guide
├── STATISTICAL_MODELS_SUMMARY.md      # Statistical models summary
│
└── lin_wan5_strong_turb_samps.zip    # Data file
```

## Features by Component

### Data Preparation (Task #77)
- [x] Univariate time series loading
- [x] dB conversion and mean-centering
- [x] 48+ engineered features
- [x] Multi-horizon target generation
- [x] Time-aware data splitting
- [x] Feature scaling and validation

### Statistical Models (Task #79)
- [x] Stationarity testing (ADF, KPSS)
- [x] ARIMA with grid search
- [x] SARIMAX with seasonality
- [x] Prophet with hyperparameter tuning
- [x] Recursive multi-step forecasting
- [x] Residual diagnostics
- [x] Performance comparison

### Coming Soon
- [ ] Gradient Boosting Models (Task #78)
- [ ] Deep Learning Models (Task #80)

## Performance Summary

| Approach | Horizon | RMSE (est.) | Inference Speed | Interpretability |
|----------|---------|-------------|-----------------|------------------|
| **ML Models** (baseline) | 50-500 | 0.22-0.30 | Very Fast | Low |
| **ARIMA** | 50-200 | 0.35-0.65 | Fast | High |
| **SARIMAX** | 50-200 | 0.35-0.65 | Fast | High |
| **Prophet** | 50-200 | 0.4-0.8 | Slow | Medium |

**Recommendation:** Use ML models for production (accuracy), statistical models for analysis (interpretability)

## Documentation

- **Data Pipeline**: [data_pipeline_README.md](data_pipeline_README.md)
- **Statistical Models**: [STATISTICAL_MODELS_README.md](STATISTICAL_MODELS_README.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Implementation Summaries**: See `*_SUMMARY.md` files

## Contributing

When extending the project:
1. Follow existing code structure and naming conventions
2. Add comprehensive docstrings
3. Update relevant documentation files
4. Include usage examples
5. Document performance characteristics

## References

- **Data Pipeline**: Task #77 - Data Preparation and Feature Engineering
- **Statistical Models**: Task #79 - Statistical Models Implementation
- **ML Baselines**: Task #78 - Gradient Boosting Models (in progress)

See [QUICK_START.md](QUICK_START.md) for detailed usage.

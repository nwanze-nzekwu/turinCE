# FSO Data Preparation Pipeline - Quick Start Guide

## Installation

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

## 30-Second Quick Start

```python
from config import get_config
from data_preparation import prepare_dataset

# Prepare dataset for strong turbulence with all features
datasets = prepare_dataset('strong')

# Get training data for 0.5ms prediction horizon
train_X, train_y = datasets[5]['train']
val_X, val_y = datasets[5]['val']
test_X, test_y = datasets[5]['test']

# Train your model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X, train_y)

# Evaluate
predictions = model.predict(test_X)
```

## What You Get

### 5 Prediction Horizons
- **5 samples** (0.5 ms) - Ultra-short term
- **50 samples** (5 ms) - Short term  
- **100 samples** (10 ms) - Medium term
- **200 samples** (20 ms) - Long term
- **500 samples** (50 ms) - Very long term

### 48+ Features (Configurable)
1. **10 baseline features**: Lagged differences (lags 5-14 for latency=5)
2. **16 rolling statistics**: Mean, std, min, max over 4 windows
3. **3 EMA features**: Exponential moving averages (α=0.1, 0.3, 0.5)
4. **4 ACF features**: Autocorrelations at lags 1, 5, 10, 20
5. **12 spectral features**: FFT-based frequency analysis
6. **3 decomposition features**: Trend, residual, residual std

### Data Splits (Time-Aware)
- **70% train**: 700K+ samples (50K minimum)
- **15% validation**: 150K+ samples
- **15% test**: 150K+ samples

## Common Use Cases

### 1. Train Baseline Model
```python
from data_preparation import prepare_dataset
from sklearn.linear_model import LinearRegression

datasets = prepare_dataset('strong')
train_X, train_y = datasets[5]['train']
test_X, test_y = datasets[5]['test']

model = LinearRegression()
model.fit(train_X, train_y)
print(f"Score: {model.score(test_X, test_y):.4f}")
```

### 2. Compare Multiple Horizons
```python
from sklearn.metrics import mean_squared_error
import numpy as np

datasets = prepare_dataset('strong')

for latency in [5, 50, 100, 200, 500]:
    train_X, train_y = datasets[latency]['train']
    test_X, test_y = datasets[latency]['test']
    
    model = LinearRegression()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_y, pred))
    
    print(f"Latency {latency:3d} ({latency/10:.1f}ms): RMSE = {rmse:.4f}")
```

### 3. Custom Configuration (Faster)
```python
from config import ExperimentConfig
from data_preparation import prepare_dataset

config = ExperimentConfig()
config.turbulence.max_samples = 100_000  # Smaller dataset
config.prediction.latencies = [5, 50]     # Fewer horizons
config.features.enable_spectral = False   # Disable FFT
config.data_split.min_train_samples = 10_000

datasets = prepare_dataset('strong', config=config)
```

### 4. Feature Analysis
```python
datasets = prepare_dataset('strong')
feature_names = datasets[5]['feature_names']

print(f"Total features: {len(feature_names)}")

# Group by type
lagged = [f for f in feature_names if 'lag' in f]
rolling = [f for f in feature_names if 'rolling' in f]
ema = [f for f in feature_names if 'ema' in f]

print(f"Lagged: {len(lagged)}, Rolling: {len(rolling)}, EMA: {len(ema)}")
```

### 5. Switch Turbulence Conditions
```python
# Strong turbulence
datasets_strong = prepare_dataset('strong')

# Moderate turbulence (when available)
# datasets_moderate = prepare_dataset('moderate')

# Weak turbulence (when available)
# datasets_weak = prepare_dataset('weak')
```

## File Structure

```
.
├── config.py                  # Configuration system ⚙️
├── data_preparation.py        # Core pipeline 🔧
├── data_pipeline_README.md    # Full documentation 📖
├── example_usage.py           # Complete examples 💡
├── test_pipeline.py           # Verification tests ✅
├── QUICK_START.md            # This file 🚀
└── IMPLEMENTATION_SUMMARY.md  # Implementation details 📋
```

## Testing

Run verification tests:
```bash
python test_pipeline.py
```

Run complete examples:
```bash
python example_usage.py
```

## Configuration Options

### Disable Features
```python
config.features.enable_rolling_stats = False
config.features.enable_ema = False
config.features.enable_acf = False
config.features.enable_spectral = False
config.features.enable_decomposition = False
```

### Adjust Windows
```python
config.features.rolling_window_sizes = [5, 10]  # Fewer
config.features.ema_alphas = [0.3]              # Fewer
config.features.acf_lags = [1, 5]               # Fewer
```

### Change Split Ratios
```python
config.data_split.train_ratio = 0.80
config.data_split.val_ratio = 0.10
config.data_split.test_ratio = 0.10
```

## Performance Tips

### Speed Up Processing
```python
config.turbulence.max_samples = 100_000      # Use fewer samples
config.prediction.latencies = [5, 50]         # Fewer horizons
config.features.enable_spectral = False       # Disable FFT (slow)
config.features.enable_acf = False            # Disable ACF (slow)
```

### Reduce Memory
```python
config.turbulence.max_samples = 500_000       # Smaller dataset
config.features.rolling_window_sizes = [5, 10] # Fewer features
```

## Troubleshooting

### "FileNotFoundError"
- Ensure `lin_wan5_strong_turb_samps.zip` is in current directory
- Check file permissions

### "Memory Error"
- Reduce `max_samples`
- Disable some features
- Process one horizon at a time

### "Insufficient training samples"
- Reduce `min_train_samples`
- Increase `max_samples`

### "Taking too long"
- Reduce `max_samples`
- Disable spectral and ACF features
- Reduce number of horizons

## Next Steps

1. ✅ Run `test_pipeline.py` to verify installation
2. 📖 Read `data_pipeline_README.md` for details
3. 💡 Check `example_usage.py` for advanced examples
4. 🔧 Customize `config.py` for your needs
5. 🚀 Start training models!

## Key Functions

### Data Loading
```python
from data_preparation import load_turbulence_data
data, metadata = load_turbulence_data('strong')
```

### Feature Engineering
```python
from data_preparation import engineer_features
features_df = engineer_features(data, latency=5)
```

### Complete Pipeline
```python
from data_preparation import prepare_dataset
datasets = prepare_dataset('strong')
```

## Help & Documentation

- **Full API**: `data_pipeline_README.md`
- **Examples**: `example_usage.py`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Test**: `test_pipeline.py`

## Support

For issues or questions, check the documentation files or contact the development team.

---

**Ready to go?** Run this to get started:
```bash
python -c "from config import get_config, print_config_summary; print_config_summary(get_config('strong'))"
```

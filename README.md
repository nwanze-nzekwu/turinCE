# turinCE - FSO Channel Power Estimation

## New: Data Preparation Pipeline ✨

A comprehensive data preparation pipeline has been implemented with 48+ features and multi-horizon support.

**Quick Start**: See [QUICK_START.md](QUICK_START.md)  
**Full Documentation**: See [data_pipeline_README.md](data_pipeline_README.md)  
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

### Installation
```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

### Verify
```bash
python test_pipeline.py
```

See [QUICK_START.md](QUICK_START.md) for detailed usage.

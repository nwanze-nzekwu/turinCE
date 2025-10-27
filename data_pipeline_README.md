# FSO Channel Power Estimation - Data Preparation Pipeline

## Overview

This data preparation pipeline provides a comprehensive, modular framework for processing FSO (Free Space Optical) channel power time series data and engineering features for multi-horizon prediction tasks.

## Features

### Data Processing
- **Flexible data loading**: Supports strong, moderate, and weak turbulence conditions
- **Automatic preprocessing**: dB conversion and mean-centering
- **ZIP file extraction**: Direct loading from compressed .mat files
- **Memory efficient**: Configurable sample limits

### Feature Engineering

#### Baseline Features
- **Lagged differences**: Configurable number of taps and lag ranges
- Matches existing baseline implementation with nTaps=10

#### Extended Features
- **Rolling window statistics**: mean, std, min, max (multiple window sizes)
- **Exponential Moving Averages (EMA)**: Multiple decay parameters (α = 0.1, 0.3, 0.5)
- **Autocorrelation features (ACF)**: Multiple lag values (1, 5, 10, 20)
- **Spectral features (FFT)**: 
  - Dominant frequency components
  - Spectral power in key bands
  - Spectral centroid
- **Trend/seasonality decomposition**: 
  - Moving average trend extraction
  - Residual signal features

### Multi-Horizon Support
- **5 prediction horizons**: 5, 50, 100, 200, 500 samples (0.5ms to 50ms @ 10kHz)
- **Independent feature generation**: Each horizon gets appropriate feature windows
- **No data leakage**: Strict temporal ordering maintained

### Data Splitting
- **Time-aware splitting**: Preserves temporal order
- **70/15/15 split**: Train/Validation/Test
- **Configurable minimum**: Default 50,000+ training samples (up from 10,000)
- **Sufficient samples**: Ensures adequate data for all horizons

### Code Quality
- **Modular design**: Separate functions for each feature type
- **Type hints**: Full type annotations for clarity
- **Comprehensive documentation**: Detailed docstrings
- **Configuration-driven**: Easy experimentation via config files
- **Validation**: Automatic checks for NaN, inf, and shape issues

## Installation

### Dependencies

```bash
pip install numpy pandas scipy scikit-learn statsmodels
```

### Required packages:
- `numpy`: Array operations
- `pandas`: DataFrame manipulation
- `scipy`: Signal processing, I/O
- `scikit-learn`: Preprocessing, scaling
- `statsmodels`: Time series decomposition

## Quick Start

### Basic Usage

```python
from config import get_config, print_config_summary
from data_preparation import prepare_dataset

# Create configuration for strong turbulence
config = get_config('strong')
print_config_summary(config)

# Prepare complete dataset for all horizons
datasets = prepare_dataset('strong', config=config)

# Access data for specific horizon
latency = 5  # 0.5ms prediction horizon
train_X, train_y = datasets[latency]['train']
val_X, val_y = datasets[latency]['val']
test_X, test_y = datasets[latency]['test']

print(f"Training samples: {len(train_X):,}")
print(f"Number of features: {len(datasets[latency]['feature_names'])}")
```

### Custom Configuration

```python
from config import ExperimentConfig

# Create custom configuration
config = ExperimentConfig()
config.current_condition = 'strong'
config.turbulence.max_samples = 500_000  # Use fewer samples
config.prediction.latencies = [5, 50, 100]  # Fewer horizons
config.features.enable_spectral = False  # Disable FFT features
config.data_split.min_train_samples = 30_000  # Reduce minimum

datasets = prepare_dataset('strong', config=config)
```

### Loading Data Only

```python
from data_preparation import load_turbulence_data

# Load and preprocess data
data, metadata = load_turbulence_data('strong')

print(f"Loaded {len(data):,} samples")
print(f"Duration: {metadata['duration_sec']:.1f} seconds")
print(f"Mean: {metadata['mean_db']:.2f} dB, Std: {metadata['std_db']:.2f} dB")
```

### Feature Engineering Only

```python
from data_preparation import engineer_features

# Generate features for single horizon
features_df = engineer_features(data, latency=5, config=config)

print(f"Generated {len(features_df.columns)} features")
print(f"Valid samples: {len(features_df):,}")
```

## Configuration System

The pipeline uses a dataclass-based configuration system organized into modules:

### TurbulenceConfig
- Data file mappings
- Turbulence conditions
- Sampling parameters

### PredictionConfig
- Prediction horizons (latencies)
- Number of lagged features (taps)
- Differencing options

### FeatureConfig
- Enable/disable feature groups
- Rolling window parameters
- EMA decay factors
- ACF lags
- FFT settings
- Decomposition parameters
- Scaling options

### DataSplitConfig
- Split ratios
- Minimum sample requirements
- Random seed
- Time-aware splitting

### ValidationConfig
- NaN checking
- Infinite value detection
- Shape validation

## Architecture

### Module Structure

```
config.py                 # Configuration classes and defaults
data_preparation.py       # Core data pipeline
data_pipeline_README.md   # This documentation
example_usage.py          # Usage examples
```

### Key Functions

#### `load_turbulence_data(condition, config, data_dir)`
Loads and preprocesses turbulence data from ZIP files.

**Returns**: `(data_array, metadata_dict)`

#### `engineer_features(data, latency, config)`
Generates all features for a specific prediction horizon.

**Returns**: `DataFrame` with features and target

#### `create_splits(features, target_col, config)`
Creates time-aware train/val/test splits.

**Returns**: `Dict` with splits and metadata

#### `prepare_dataset(condition, config, data_dir, validate)`
Complete end-to-end pipeline (recommended entry point).

**Returns**: `Dict[latency -> dataset]` with all splits for all horizons

## Feature Descriptions

### Baseline Features (Lagged Differences)
- `OptPow_diff_lag{i}`: Differenced power at lag i
- Lags range from `latency` to `latency + n_taps - 1`
- Example: For latency=5, n_taps=10: lags [5, 6, 7, ..., 14]

### Rolling Statistics
- `OptPow_rolling_{stat}_{window}`: Rolling statistic over window
- Stats: mean, std, min, max
- Windows: 5, 10, 20, 50 samples

### Exponential Moving Average
- `OptPow_ema_{alpha}`: EMA with decay factor alpha
- Alpha values: 0.1, 0.3, 0.5

### Autocorrelation
- `OptPow_acf_lag{i}`: Autocorrelation at lag i
- Lags: 1, 5, 10, 20

### Spectral Features
- `OptPow_spectral_power_total`: Total spectral power
- `OptPow_spectral_centroid`: Weighted mean frequency
- `OptPow_spectral_freq{i}`: i-th dominant frequency
- `OptPow_spectral_power{i}`: Power at i-th dominant frequency

### Decomposition
- `OptPow_trend`: Trend component (moving average)
- `OptPow_residual`: Detrended signal
- `OptPow_residual_std`: Rolling std of residual

### Target Variable
- `target_{latency}`: Power difference over latency horizon
- Computed as: `OptPow[t] - OptPow[t-latency]`

## Data Validation

The pipeline includes automatic validation:
- ✓ NaN detection and reporting
- ✓ Infinite value detection
- ✓ Shape consistency checks
- ✓ Range validation

Validation runs automatically unless disabled:
```python
datasets = prepare_dataset('strong', validate=False)  # Skip validation
```

## Memory Considerations

For 1M samples with full feature set:
- **Expected memory**: ~2-4 GB per horizon
- **Total for 5 horizons**: ~10-20 GB
- **Optimization**: Reduce enabled features or use fewer samples

### Memory optimization:
```python
config.turbulence.max_samples = 500_000  # Reduce samples
config.features.enable_spectral = False  # Disable expensive features
config.features.rolling_window_sizes = [5, 10]  # Fewer windows
```

## Extensibility

### Adding New Turbulence Conditions

Edit `config.py`:
```python
DATA_FILES: Dict[str, Dict[str, str]] = {
    'new_condition': {
        'file': 'new_data.zip',
        'mat_file': 'new_data.mat',
        'variable': 'new_var_name'
    }
}
```

Usage:
```python
datasets = prepare_dataset('new_condition')
```

### Adding New Features

Create a new function in `data_preparation.py`:
```python
def add_custom_features(df, base_col='OptPow'):
    df_copy = df.copy()
    # Add your features
    df_copy['custom_feature'] = compute_custom_feature(df_copy[base_col])
    return df_copy
```

Integrate in `engineer_features()`:
```python
if config.features.enable_custom:
    df = add_custom_features(df)
```

### Adding New Horizons

Update `config.py`:
```python
@dataclass
class PredictionConfig:
    latencies: List[int] = field(default_factory=lambda: [5, 10, 50, 100, 200, 500, 1000])
```

## Performance

### Approximate Processing Times (1M samples, all features)

- Data loading: ~5-10 seconds
- Feature engineering per horizon: ~30-60 seconds
- Total for 5 horizons: ~3-5 minutes
- Splitting and scaling: ~5-10 seconds

### Optimization Tips

1. **Disable unused features**: Set `enable_*=False` in config
2. **Reduce window sizes**: Use smaller FFT/rolling windows
3. **Fewer horizons**: Only generate needed latencies
4. **Parallel processing**: Process horizons in parallel (future enhancement)

## Output Structure

The `prepare_dataset()` function returns:

```python
{
    5: {  # latency in samples
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'scaler': StandardScaler(),
        'feature_names': ['feature1', 'feature2', ...],
        'metadata': {...}
    },
    50: {...},
    100: {...},
    200: {...},
    500: {...}
}
```

## Troubleshooting

### FileNotFoundError
- Ensure `lin_wan5_strong_turb_samps.zip` is in project directory
- Check `data_dir` parameter points to correct location

### Memory Error
- Reduce `max_samples` in config
- Disable some feature groups
- Process horizons separately

### Insufficient Training Samples
- Reduce `min_train_samples` in config
- Increase `max_samples` to load more data

### NaN Values in Features
- Expected for initial rows due to lagging/windowing
- Automatically dropped in `engineer_features()`
- Verify sufficient samples remain after dropping

## Best Practices

1. **Start with default config**: Use `get_config()` for initial experiments
2. **Profile memory**: Monitor RAM usage for large datasets
3. **Validate consistently**: Keep validation enabled during development
4. **Save processed data**: Cache feature matrices to disk for repeated use
5. **Document changes**: Update config when modifying feature sets
6. **Version control**: Track config changes alongside code

## Examples

See `example_usage.py` for complete examples including:
- Basic pipeline usage
- Custom configuration
- Feature analysis
- Model training integration
- Cross-condition comparison

## Citation

If you use this pipeline, please cite:
```
FSO Channel Power Estimation Data Preparation Pipeline
TurinTech FSO Project, 2024
```

## Support

For issues, questions, or contributions, please contact the development team.

## License

[Your license here]

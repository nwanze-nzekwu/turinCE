"""
Configuration module for FSO Channel Power Estimation.

This module contains all configuration parameters for turbulence conditions,
prediction horizons, feature engineering, and data splitting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class TurbulenceConfig:
    """Configuration for turbulence data loading."""
    
    # Available turbulence conditions
    STRONG: str = 'strong'
    MODERATE: str = 'moderate'
    WEAK: str = 'weak'
    
    # Data file mapping
    DATA_FILES: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'strong': {
            'file': 'lin_wan5_strong_turb_samps.zip',
            'mat_file': 'lin_wan5_strong_turb_samps.mat',
            'variable': 'lin_wan5_s_dat'
        },
        'moderate': {
            'file': 'lin_wan5_mod_turb_samps.zip',
            'mat_file': 'lin_wan5_mod_turb_samps.mat',
            'variable': 'lin_wan5_m_dat'
        },
        'weak': {
            'file': 'lin_wan5_weak_turb_samps.zip',
            'mat_file': 'lin_wan5_weak_turb_samps.mat',
            'variable': 'lin_wan5_w_dat'
        }
    })
    
    # Maximum samples to load (None for all)
    max_samples: int = 1_000_000
    
    # Sampling rate in Hz
    sampling_rate: int = 10_000


@dataclass
class PredictionConfig:
    """Configuration for prediction horizons and targets."""
    
    # Prediction horizons in samples (0.5ms to 50ms at 10kHz)
    latencies: List[int] = field(default_factory=lambda: [5, 50, 100, 200, 500])
    
    # Number of lagged features (taps)
    n_taps: int = 10
    
    # Whether to use differenced features
    use_diff: bool = True


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Baseline features: lagged differences
    # Lags are from latency to latency+n_taps (e.g., 5-14 for latency=5, n_taps=10)
    enable_lagged_diffs: bool = True
    
    # Rolling window statistics
    enable_rolling_stats: bool = True
    rolling_window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rolling_stats: List[str] = field(default_factory=lambda: ['mean', 'std', 'min', 'max'])
    
    # Exponential Moving Average (EMA)
    enable_ema: bool = True
    ema_alphas: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    
    # Autocorrelation features
    enable_acf: bool = True
    acf_lags: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Spectral features (FFT)
    enable_spectral: bool = True
    fft_window_size: int = 100  # Window size for FFT computation
    n_top_frequencies: int = 5  # Number of dominant frequencies to extract
    
    # Trend/seasonality decomposition
    enable_decomposition: bool = True
    decomp_window_size: int = 50  # Window for trend extraction
    
    # Feature scaling
    enable_scaling: bool = True
    scaling_method: str = 'standard'  # 'standard', 'minmax', or 'robust'


@dataclass
class DataSplitConfig:
    """Configuration for train/validation/test splitting."""
    
    # Split ratios (must sum to ~1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Minimum training samples (increased from 10,000)
    min_train_samples: int = 50_000
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Time-aware splitting (maintains temporal order)
    time_aware: bool = True


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    
    # Check for NaN values
    check_nan: bool = True
    
    # Check for infinite values
    check_inf: bool = True
    
    # Check for correct shapes
    check_shapes: bool = True
    
    # Tolerance for floating point comparisons
    float_tolerance: float = 1e-10


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Current turbulence condition to use
    current_condition: str = 'strong'
    
    def get_data_config(self) -> Dict[str, str]:
        """Get data file configuration for current turbulence condition."""
        return self.turbulence.DATA_FILES[self.current_condition]
    
    def get_latency_time_ms(self, latency_samples: int) -> float:
        """Convert latency in samples to milliseconds."""
        return (latency_samples / self.turbulence.sampling_rate) * 1000


# Default configuration instance
default_config = ExperimentConfig()


def get_config(condition: str = 'strong') -> ExperimentConfig:
    """
    Get a configuration instance for a specific turbulence condition.
    
    Args:
        condition: Turbulence condition ('strong', 'moderate', or 'weak')
        
    Returns:
        ExperimentConfig instance configured for the specified condition
    """
    config = ExperimentConfig()
    config.current_condition = condition
    return config


def print_config_summary(config: ExperimentConfig) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration instance to summarize
    """
    print("=" * 60)
    print("FSO Channel Power Estimation - Configuration Summary")
    print("=" * 60)
    print(f"Turbulence Condition: {config.current_condition}")
    print(f"Max Samples: {config.turbulence.max_samples:,}")
    print(f"Sampling Rate: {config.turbulence.sampling_rate:,} Hz")
    print(f"\nPrediction Horizons (samples): {config.prediction.latencies}")
    print(f"Prediction Horizons (ms): {[config.get_latency_time_ms(l) for l in config.prediction.latencies]}")
    print(f"Number of Taps: {config.prediction.n_taps}")
    print(f"\nFeatures Enabled:")
    print(f"  - Lagged Differences: {config.features.enable_lagged_diffs}")
    print(f"  - Rolling Statistics: {config.features.enable_rolling_stats}")
    print(f"  - Exponential Moving Average: {config.features.enable_ema}")
    print(f"  - Autocorrelation: {config.features.enable_acf}")
    print(f"  - Spectral (FFT): {config.features.enable_spectral}")
    print(f"  - Decomposition: {config.features.enable_decomposition}")
    print(f"\nData Split:")
    print(f"  - Train: {config.data_split.train_ratio:.0%}")
    print(f"  - Validation: {config.data_split.val_ratio:.0%}")
    print(f"  - Test: {config.data_split.test_ratio:.0%}")
    print(f"  - Min Training Samples: {config.data_split.min_train_samples:,}")
    print("=" * 60)

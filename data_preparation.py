"""
Data Preparation Module for FSO Channel Power Estimation.

This module provides comprehensive data processing, feature engineering, and
splitting capabilities for univariate FSO power time series prediction.
"""

import os
import zipfile
import io
from typing import Tuple, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import STL

from config import ExperimentConfig


def pow2db(x: np.ndarray) -> np.ndarray:
    """
    Convert power to decibels (dB).
    
    Args:
        x: Power values (linear scale)
        
    Returns:
        Power in dB scale (10*log10(x))
    """
    return 10 * np.log10(x)


def load_turbulence_data(
    condition: str = 'strong',
    config: Optional[ExperimentConfig] = None,
    data_dir: str = '.'
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Load and preprocess FSO turbulence data from zip file.
    
    This function:
    1. Extracts .mat file from zip archive
    2. Loads specified turbulence variable
    3. Applies dB conversion
    4. Mean-centers the data
    
    Args:
        condition: Turbulence condition ('strong', 'moderate', or 'weak')
        config: Configuration object (uses default if None)
        data_dir: Directory containing data files
        
    Returns:
        Tuple of:
            - Preprocessed data array (mean-centered dB scale)
            - Metadata dictionary with info about the data
            
    Raises:
        FileNotFoundError: If data file doesn't exist
        KeyError: If variable not found in .mat file
    """
    if config is None:
        from config import get_config
        config = get_config(condition)
    
    # Get data file configuration
    data_config = config.turbulence.DATA_FILES[condition]
    zip_path = os.path.join(data_dir, data_config['file'])
    mat_filename = data_config['mat_file']
    var_name = data_config['variable']
    
    # Check if file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Data file not found: {zip_path}\n"
            f"Expected turbulence data for condition: {condition}"
        )
    
    # Extract and load .mat file from zip
    print(f"Loading {condition} turbulence data from {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Read .mat file from zip
        with zip_ref.open(mat_filename) as mat_file:
            mat_data = sio.loadmat(io.BytesIO(mat_file.read()))
    
    # Extract the variable
    if var_name not in mat_data:
        raise KeyError(
            f"Variable '{var_name}' not found in {mat_filename}. "
            f"Available variables: {list(mat_data.keys())}"
        )
    
    raw_data = mat_data[var_name].flatten()
    
    # Limit samples if specified
    max_samples = config.turbulence.max_samples
    if max_samples is not None and max_samples < len(raw_data):
        raw_data = raw_data[:max_samples]
    
    # Preprocessing: Convert to dB and mean-center
    data_db = pow2db(raw_data)
    data_centered = data_db - np.mean(data_db)
    
    # Create metadata
    metadata = {
        'condition': condition,
        'n_samples': len(data_centered),
        'sampling_rate': config.turbulence.sampling_rate,
        'duration_sec': len(data_centered) / config.turbulence.sampling_rate,
        'mean_db': np.mean(data_db),
        'std_db': np.std(data_centered),
        'min_db': np.min(data_centered),
        'max_db': np.max(data_centered),
        'raw_file': zip_path,
        'variable': var_name
    }
    
    print(f"Loaded {len(data_centered):,} samples "
          f"({metadata['duration_sec']:.1f} seconds)")
    print(f"Stats - Mean: {metadata['mean_db']:.2f} dB, "
          f"Std: {metadata['std_db']:.2f} dB")
    
    return data_centered, metadata


def create_lagged_features(
    data: np.ndarray,
    latency: int,
    n_taps: int = 10,
    use_diff: bool = True
) -> pd.DataFrame:
    """
    Create baseline lagged difference features.
    
    Features are created from lags [latency, latency+n_taps).
    For example, with latency=5 and n_taps=10, lags are [5, 6, 7, ..., 14].
    
    Args:
        data: Input time series (1D array)
        latency: Prediction horizon in samples
        n_taps: Number of lagged features
        use_diff: If True, use differenced data; if False, use raw data
        
    Returns:
        DataFrame with lagged features
    """
    df = pd.DataFrame({'OptPow': data})
    
    if use_diff:
        # Create differenced series
        df['OptPow_diff'] = df['OptPow'].diff()
        base_col = 'OptPow_diff'
    else:
        base_col = 'OptPow'
    
    # Create lagged features
    for lag in range(latency, latency + n_taps):
        col_name = f'{base_col}_lag{lag}'
        df[col_name] = df[base_col].shift(lag)
    
    return df


def add_rolling_statistics(
    df: pd.DataFrame,
    window_sizes: List[int],
    stats: List[str] = ['mean', 'std', 'min', 'max'],
    base_col: str = 'OptPow'
) -> pd.DataFrame:
    """
    Add rolling window statistics as features.
    
    Args:
        df: DataFrame with base time series
        window_sizes: List of window sizes to compute statistics over
        stats: List of statistics to compute ('mean', 'std', 'min', 'max')
        base_col: Column to compute statistics on
        
    Returns:
        DataFrame with added rolling statistics columns
    """
    df_copy = df.copy()
    
    for window in window_sizes:
        rolling = df_copy[base_col].rolling(window=window, min_periods=1)
        
        if 'mean' in stats:
            df_copy[f'{base_col}_rolling_mean_{window}'] = rolling.mean()
        if 'std' in stats:
            df_copy[f'{base_col}_rolling_std_{window}'] = rolling.std()
        if 'min' in stats:
            df_copy[f'{base_col}_rolling_min_{window}'] = rolling.min()
        if 'max' in stats:
            df_copy[f'{base_col}_rolling_max_{window}'] = rolling.max()
    
    return df_copy


def add_exponential_moving_average(
    df: pd.DataFrame,
    alphas: List[float],
    base_col: str = 'OptPow'
) -> pd.DataFrame:
    """
    Add exponential moving average (EMA) features.
    
    Args:
        df: DataFrame with base time series
        alphas: List of smoothing factors (0 < alpha <= 1)
        base_col: Column to compute EMA on
        
    Returns:
        DataFrame with added EMA columns
    """
    df_copy = df.copy()
    
    for alpha in alphas:
        col_name = f'{base_col}_ema_{alpha}'
        df_copy[col_name] = df_copy[base_col].ewm(alpha=alpha, adjust=False).mean()
    
    return df_copy


def add_autocorrelation_features(
    df: pd.DataFrame,
    lags: List[int],
    base_col: str = 'OptPow',
    window_size: int = 100
) -> pd.DataFrame:
    """
    Add autocorrelation features at specified lags.
    
    Uses a rolling window approach to compute local autocorrelations.
    
    Args:
        df: DataFrame with base time series
        lags: List of lags to compute autocorrelation at
        base_col: Column to compute autocorrelation on
        window_size: Window size for computing rolling autocorrelation
        
    Returns:
        DataFrame with added autocorrelation columns
    """
    df_copy = df.copy()
    data = df_copy[base_col].values
    
    for lag in lags:
        acf_values = np.full(len(data), np.nan)
        
        # Compute ACF using rolling window
        for i in range(window_size, len(data)):
            window = data[i-window_size:i]
            if len(window) > lag:
                # Compute correlation between window and lagged window
                lagged = np.roll(window, lag)
                # Only use valid portion (not affected by roll-over)
                valid_len = len(window) - lag
                if valid_len > 1:
                    corr = np.corrcoef(window[:valid_len], lagged[:valid_len])[0, 1]
                    acf_values[i] = corr if not np.isnan(corr) else 0.0
        
        # Fill initial NaN values with 0
        acf_values[:window_size] = 0.0
        df_copy[f'{base_col}_acf_lag{lag}'] = acf_values
    
    return df_copy


def add_spectral_features(
    df: pd.DataFrame,
    base_col: str = 'OptPow',
    window_size: int = 100,
    n_top_frequencies: int = 5,
    sampling_rate: int = 10000
) -> pd.DataFrame:
    """
    Add spectral features from FFT analysis.
    
    Uses rolling window FFT to extract dominant frequency components.
    
    Args:
        df: DataFrame with base time series
        base_col: Column to compute FFT on
        window_size: Window size for FFT computation
        n_top_frequencies: Number of dominant frequencies to extract
        sampling_rate: Sampling rate in Hz
        
    Returns:
        DataFrame with added spectral features
    """
    df_copy = df.copy()
    data = df_copy[base_col].values
    n = len(data)
    
    # Initialize feature arrays
    spectral_features = {
        f'{base_col}_spectral_power_total': np.full(n, np.nan),
        f'{base_col}_spectral_centroid': np.full(n, np.nan),
    }
    
    for i in range(n_top_frequencies):
        spectral_features[f'{base_col}_spectral_freq{i+1}'] = np.full(n, np.nan)
        spectral_features[f'{base_col}_spectral_power{i+1}'] = np.full(n, np.nan)
    
    # Compute spectral features using rolling window
    for i in range(window_size, n):
        window = data[i-window_size:i]
        
        # Apply Hann window to reduce spectral leakage
        windowed_data = window * np.hanning(len(window))
        
        # Compute FFT
        fft_values = np.fft.rfft(windowed_data)
        fft_power = np.abs(fft_values) ** 2
        fft_freqs = np.fft.rfftfreq(len(window), d=1.0/sampling_rate)
        
        # Total spectral power
        spectral_features[f'{base_col}_spectral_power_total'][i] = np.sum(fft_power)
        
        # Spectral centroid (weighted mean frequency)
        if np.sum(fft_power) > 0:
            centroid = np.sum(fft_freqs * fft_power) / np.sum(fft_power)
            spectral_features[f'{base_col}_spectral_centroid'][i] = centroid
        
        # Top N dominant frequencies and their powers
        top_indices = np.argsort(fft_power)[-n_top_frequencies:][::-1]
        for j, idx in enumerate(top_indices):
            spectral_features[f'{base_col}_spectral_freq{j+1}'][i] = fft_freqs[idx]
            spectral_features[f'{base_col}_spectral_power{j+1}'][i] = fft_power[idx]
    
    # Fill initial NaN values with forward fill
    for key, values in spectral_features.items():
        values[:window_size] = values[window_size]
        df_copy[key] = values
    
    return df_copy


def add_decomposition_features(
    df: pd.DataFrame,
    base_col: str = 'OptPow',
    window_size: int = 50
) -> pd.DataFrame:
    """
    Add trend and residual features using moving average decomposition.
    
    For non-seasonal time series, uses simple moving average for trend extraction.
    
    Args:
        df: DataFrame with base time series
        base_col: Column to decompose
        window_size: Window size for trend extraction
        
    Returns:
        DataFrame with added decomposition features (trend, residual)
    """
    df_copy = df.copy()
    data = df_copy[base_col]
    
    # Compute trend using centered moving average
    trend = data.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # Compute residual (detrended signal)
    residual = data - trend
    
    # Add features
    df_copy[f'{base_col}_trend'] = trend
    df_copy[f'{base_col}_residual'] = residual
    df_copy[f'{base_col}_residual_std'] = residual.rolling(
        window=window_size, min_periods=1
    ).std()
    
    return df_copy


def engineer_features(
    data: np.ndarray,
    latency: int,
    config: Optional[ExperimentConfig] = None
) -> pd.DataFrame:
    """
    Generate all features for a given prediction horizon.
    
    This is the main feature engineering function that orchestrates all
    feature generation based on configuration.
    
    Args:
        data: Input time series (1D array, mean-centered dB scale)
        latency: Prediction horizon in samples
        config: Configuration object (uses default if None)
        
    Returns:
        DataFrame with all engineered features
    """
    if config is None:
        from config import default_config
        config = default_config
    
    print(f"\nEngineering features for latency={latency} samples "
          f"({config.get_latency_time_ms(latency):.2f} ms)...")
    
    # Start with baseline lagged features
    df = create_lagged_features(
        data,
        latency=latency,
        n_taps=config.prediction.n_taps,
        use_diff=config.prediction.use_diff
    )
    
    # Add rolling statistics
    if config.features.enable_rolling_stats:
        df = add_rolling_statistics(
            df,
            window_sizes=config.features.rolling_window_sizes,
            stats=config.features.rolling_stats,
            base_col='OptPow'
        )
    
    # Add exponential moving averages
    if config.features.enable_ema:
        df = add_exponential_moving_average(
            df,
            alphas=config.features.ema_alphas,
            base_col='OptPow'
        )
    
    # Add autocorrelation features
    if config.features.enable_acf:
        df = add_autocorrelation_features(
            df,
            lags=config.features.acf_lags,
            base_col='OptPow',
            window_size=100
        )
    
    # Add spectral features
    if config.features.enable_spectral:
        df = add_spectral_features(
            df,
            base_col='OptPow',
            window_size=config.features.fft_window_size,
            n_top_frequencies=config.features.n_top_frequencies,
            sampling_rate=config.turbulence.sampling_rate
        )
    
    # Add decomposition features
    if config.features.enable_decomposition:
        df = add_decomposition_features(
            df,
            base_col='OptPow',
            window_size=config.features.decomp_window_size
        )
    
    # Add target variable
    df[f'OptPow_lag{latency}'] = df['OptPow'].shift(latency)
    df[f'target_{latency}'] = df['OptPow'] - df[f'OptPow_lag{latency}']
    
    # Drop rows with NaN values
    initial_rows = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_rows - len(df)
    
    print(f"Generated {len(df.columns)} columns, {len(df):,} valid samples "
          f"({dropped_rows} rows dropped due to NaN)")
    
    return df


def create_multi_horizon_features(
    data: np.ndarray,
    config: Optional[ExperimentConfig] = None
) -> Dict[int, pd.DataFrame]:
    """
    Generate features for all configured prediction horizons.
    
    Args:
        data: Input time series (1D array, mean-centered dB scale)
        config: Configuration object (uses default if None)
        
    Returns:
        Dictionary mapping latency -> DataFrame with features
    """
    if config is None:
        from config import default_config
        config = default_config
    
    print("=" * 60)
    print("Multi-Horizon Feature Engineering")
    print("=" * 60)
    
    feature_dfs = {}
    
    for latency in config.prediction.latencies:
        feature_dfs[latency] = engineer_features(data, latency, config)
    
    print("\n" + "=" * 60)
    print("Feature Engineering Complete")
    print("=" * 60)
    
    return feature_dfs


def create_splits(
    features: pd.DataFrame,
    target_col: str,
    config: Optional[ExperimentConfig] = None
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Create time-aware train/validation/test splits.
    
    Maintains temporal order to prevent data leakage.
    
    Args:
        features: DataFrame with all features
        target_col: Name of target column
        config: Configuration object (uses default if None)
        
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing
        tuple of (X, y) where X is features and y is target
        
    Raises:
        ValueError: If insufficient samples for configured split
    """
    if config is None:
        from config import default_config
        config = default_config
    
    n_samples = len(features)
    
    # Calculate split indices (time-aware)
    train_end = int(n_samples * config.data_split.train_ratio)
    val_end = train_end + int(n_samples * config.data_split.val_ratio)
    
    # Check minimum training samples
    if train_end < config.data_split.min_train_samples:
        raise ValueError(
            f"Insufficient training samples: {train_end} < "
            f"{config.data_split.min_train_samples}. "
            f"Reduce min_train_samples or increase data size."
        )
    
    # Identify feature columns (exclude target and auxiliary columns)
    exclude_cols = [target_col, 'OptPow', 'OptPow_diff'] + \
                   [col for col in features.columns if col.startswith('OptPow_lag')]
    feature_cols = [col for col in features.columns if col not in exclude_cols]
    
    # Create splits
    train_X = features.iloc[:train_end][feature_cols]
    train_y = features.iloc[:train_end][target_col]
    
    val_X = features.iloc[train_end:val_end][feature_cols]
    val_y = features.iloc[train_end:val_end][target_col]
    
    test_X = features.iloc[val_end:][feature_cols]
    test_y = features.iloc[val_end:][target_col]
    
    print(f"\nData Split Summary:")
    print(f"  Train: {len(train_X):,} samples ({len(train_X)/n_samples:.1%})")
    print(f"  Val:   {len(val_X):,} samples ({len(val_X)/n_samples:.1%})")
    print(f"  Test:  {len(test_X):,} samples ({len(test_X)/n_samples:.1%})")
    print(f"  Features: {len(feature_cols)} columns")
    
    return {
        'train': (train_X, train_y),
        'val': (val_X, val_y),
        'test': (test_X, test_y),
        'feature_names': feature_cols
    }


def scale_features(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    test_X: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, any]:
    """
    Scale features using specified method.
    
    Fits scaler on training data only to prevent data leakage.
    
    Args:
        train_X: Training features
        val_X: Validation features
        test_X: Test features
        method: Scaling method ('standard', 'minmax', or 'robust')
        
    Returns:
        Tuple of (scaled_train_X, scaled_val_X, scaled_test_X, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit on training data only
    scaler.fit(train_X)
    
    # Transform all sets
    train_X_scaled = pd.DataFrame(
        scaler.transform(train_X),
        columns=train_X.columns,
        index=train_X.index
    )
    val_X_scaled = pd.DataFrame(
        scaler.transform(val_X),
        columns=val_X.columns,
        index=val_X.index
    )
    test_X_scaled = pd.DataFrame(
        scaler.transform(test_X),
        columns=test_X.columns,
        index=test_X.index
    )
    
    print(f"Features scaled using {method} scaler")
    
    return train_X_scaled, val_X_scaled, test_X_scaled, scaler


def validate_data(
    data: Union[np.ndarray, pd.DataFrame],
    config: Optional[ExperimentConfig] = None,
    name: str = "data"
) -> bool:
    """
    Validate data for NaN, infinite values, and correct shapes.
    
    Args:
        data: Data to validate (array or DataFrame)
        config: Configuration object (uses default if None)
        name: Name for reporting purposes
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if config is None:
        from config import default_config
        config = default_config
    
    if isinstance(data, pd.DataFrame):
        values = data.values
    else:
        values = data
    
    issues = []
    
    # Check for NaN
    if config.validation.check_nan:
        nan_count = np.isnan(values).sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count} NaN values")
    
    # Check for infinite values
    if config.validation.check_inf:
        inf_count = np.isinf(values).sum()
        if inf_count > 0:
            issues.append(f"Contains {inf_count} infinite values")
    
    # Check shape
    if config.validation.check_shapes:
        if values.size == 0:
            issues.append("Data is empty")
    
    if issues:
        raise ValueError(f"Validation failed for {name}:\n" + "\n".join(f"  - {i}" for i in issues))
    
    print(f"✓ Validation passed for {name}: shape={values.shape}, "
          f"range=[{values.min():.4f}, {values.max():.4f}]")
    
    return True


def prepare_dataset(
    condition: str = 'strong',
    config: Optional[ExperimentConfig] = None,
    data_dir: str = '.',
    validate: bool = True
) -> Dict[int, Dict[str, any]]:
    """
    Complete data preparation pipeline.
    
    This is the main entry point that orchestrates the entire data preparation:
    1. Load turbulence data
    2. Engineer features for all horizons
    3. Create train/val/test splits
    4. Scale features
    5. Validate data
    
    Args:
        condition: Turbulence condition ('strong', 'moderate', or 'weak')
        config: Configuration object (uses default if None)
        data_dir: Directory containing data files
        validate: Whether to run validation checks
        
    Returns:
        Dictionary mapping latency -> dataset dict with keys:
            - 'train': (X_train, y_train)
            - 'val': (X_val, y_val)
            - 'test': (X_test, y_test)
            - 'scaler': fitted scaler object
            - 'feature_names': list of feature names
            - 'metadata': data metadata
    """
    if config is None:
        from config import get_config
        config = get_config(condition)
    
    print("\n" + "=" * 60)
    print("FSO Channel Power Estimation - Data Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    data, metadata = load_turbulence_data(condition, config, data_dir)
    
    if validate:
        validate_data(data, config, name="raw data")
    
    # Step 2: Engineer features for all horizons
    feature_dfs = create_multi_horizon_features(data, config)
    
    # Step 3: Create splits and scale for each horizon
    datasets = {}
    
    for latency, features_df in feature_dfs.items():
        print(f"\n--- Processing Latency = {latency} samples ---")
        
        target_col = f'target_{latency}'
        
        # Create splits
        splits = create_splits(features_df, target_col, config)
        
        # Scale features if enabled
        if config.features.enable_scaling:
            train_X_scaled, val_X_scaled, test_X_scaled, scaler = scale_features(
                splits['train'][0],
                splits['val'][0],
                splits['test'][0],
                method=config.features.scaling_method
            )
            
            datasets[latency] = {
                'train': (train_X_scaled, splits['train'][1]),
                'val': (val_X_scaled, splits['val'][1]),
                'test': (test_X_scaled, splits['test'][1]),
                'scaler': scaler,
                'feature_names': splits['feature_names'],
                'metadata': metadata
            }
        else:
            datasets[latency] = {
                'train': splits['train'],
                'val': splits['val'],
                'test': splits['test'],
                'scaler': None,
                'feature_names': splits['feature_names'],
                'metadata': metadata
            }
        
        # Validate if enabled
        if validate:
            validate_data(datasets[latency]['train'][0], config, 
                         name=f"train features (latency={latency})")
            validate_data(datasets[latency]['train'][1], config,
                         name=f"train targets (latency={latency})")
    
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nPrepared datasets for {len(datasets)} prediction horizons:")
    for latency in sorted(datasets.keys()):
        n_train = len(datasets[latency]['train'][0])
        n_features = len(datasets[latency]['feature_names'])
        print(f"  - Latency {latency:3d} samples ({config.get_latency_time_ms(latency):5.1f} ms): "
              f"{n_train:,} training samples, {n_features} features")
    
    return datasets


if __name__ == "__main__":
    # Example usage
    from config import get_config, print_config_summary
    
    # Create configuration
    config = get_config('strong')
    print_config_summary(config)
    
    # Prepare dataset
    datasets = prepare_dataset('strong', config=config)
    
    # Access data for a specific horizon
    latency = 5
    train_X, train_y = datasets[latency]['train']
    val_X, val_y = datasets[latency]['val']
    test_X, test_y = datasets[latency]['test']
    
    print(f"\nExample: Latency = {latency} samples")
    print(f"Training set: X shape = {train_X.shape}, y shape = {train_y.shape}")
    print(f"Validation set: X shape = {val_X.shape}, y shape = {val_y.shape}")
    print(f"Test set: X shape = {test_X.shape}, y shape = {test_y.shape}")

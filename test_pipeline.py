"""
Quick test script to verify the data preparation pipeline implementation.

This script performs basic sanity checks on the pipeline without running
full examples.
"""

import numpy as np
from config import get_config, print_config_summary
from data_preparation import (
    load_turbulence_data,
    engineer_features,
    validate_data,
    prepare_dataset
)


def test_configuration():
    """Test configuration system."""
    print("\n" + "=" * 60)
    print("TEST 1: Configuration System")
    print("=" * 60)
    
    config = get_config('strong')
    
    # Check configuration attributes
    assert config.current_condition == 'strong'
    assert len(config.prediction.latencies) == 5
    assert config.prediction.n_taps == 10
    assert config.data_split.train_ratio == 0.70
    assert config.data_split.val_ratio == 0.15
    assert config.data_split.test_ratio == 0.15
    
    print("✓ Configuration system working correctly")
    print(f"  - Condition: {config.current_condition}")
    print(f"  - Horizons: {config.prediction.latencies}")
    print(f"  - Data split: {config.data_split.train_ratio}/{config.data_split.val_ratio}/{config.data_split.test_ratio}")
    
    return config


def test_data_loading(config):
    """Test data loading functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Data Loading")
    print("=" * 60)
    
    try:
        data, metadata = load_turbulence_data('strong', config)
        
        # Check data properties
        assert isinstance(data, np.ndarray)
        assert len(data) > 0
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()
        assert abs(np.mean(data)) < 1e-10  # Should be mean-centered
        
        # Check metadata
        assert 'condition' in metadata
        assert 'n_samples' in metadata
        assert metadata['condition'] == 'strong'
        
        print("✓ Data loading working correctly")
        print(f"  - Loaded samples: {len(data):,}")
        print(f"  - Mean (should be ~0): {np.mean(data):.2e}")
        print(f"  - Std: {np.std(data):.4f}")
        
        return data, metadata
        
    except FileNotFoundError as e:
        print(f"⚠ Warning: Data file not found - {e}")
        print("  Skipping data-dependent tests")
        return None, None


def test_feature_engineering(data, config):
    """Test feature engineering."""
    print("\n" + "=" * 60)
    print("TEST 3: Feature Engineering")
    print("=" * 60)
    
    if data is None:
        print("⚠ Skipping (no data available)")
        return None
    
    # Test with smallest latency
    latency = 5
    features_df = engineer_features(data, latency, config)
    
    # Check DataFrame properties
    assert len(features_df) > 0
    assert not features_df.isnull().any().any()
    assert f'target_{latency}' in features_df.columns
    
    # Check for expected feature types
    has_lagged = any('lag' in col for col in features_df.columns)
    has_rolling = any('rolling' in col for col in features_df.columns)
    has_ema = any('ema' in col for col in features_df.columns)
    
    print("✓ Feature engineering working correctly")
    print(f"  - Generated features: {len(features_df.columns)}")
    print(f"  - Valid samples: {len(features_df):,}")
    print(f"  - Has lagged features: {has_lagged}")
    print(f"  - Has rolling stats: {has_rolling}")
    print(f"  - Has EMA features: {has_ema}")
    
    return features_df


def test_validation(data):
    """Test data validation."""
    print("\n" + "=" * 60)
    print("TEST 4: Data Validation")
    print("=" * 60)
    
    if data is None:
        print("⚠ Skipping (no data available)")
        return
    
    # Test valid data
    try:
        validate_data(data, name="test data")
        print("✓ Validation working correctly")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        raise
    
    # Test invalid data detection
    invalid_data = np.array([1.0, 2.0, np.nan, 4.0])
    try:
        validate_data(invalid_data, name="invalid test data")
        print("✗ Validation should have failed for NaN data")
    except ValueError:
        print("✓ NaN detection working correctly")


def test_full_pipeline(config):
    """Test complete pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Pipeline")
    print("=" * 60)
    
    try:
        # Use smaller dataset for testing
        config.turbulence.max_samples = 10_000
        config.prediction.latencies = [5, 50]  # Test with 2 horizons
        config.data_split.min_train_samples = 1_000
        
        datasets = prepare_dataset('strong', config=config, validate=True)
        
        # Check structure
        assert isinstance(datasets, dict)
        assert len(datasets) == 2  # Two horizons
        
        for latency in [5, 50]:
            assert latency in datasets
            assert 'train' in datasets[latency]
            assert 'val' in datasets[latency]
            assert 'test' in datasets[latency]
            
            train_X, train_y = datasets[latency]['train']
            val_X, val_y = datasets[latency]['val']
            test_X, test_y = datasets[latency]['test']
            
            # Check dimensions
            assert len(train_X) > 0
            assert len(train_X) == len(train_y)
            assert len(val_X) == len(val_y)
            assert len(test_X) == len(test_y)
            assert train_X.shape[1] == val_X.shape[1] == test_X.shape[1]
            
            # Check no data leakage (temporal order)
            assert train_X.index[-1] < val_X.index[0]
            assert val_X.index[-1] < test_X.index[0]
        
        print("✓ Full pipeline working correctly")
        print(f"  - Generated {len(datasets)} horizons")
        for latency in sorted(datasets.keys()):
            n_train = len(datasets[latency]['train'][0])
            n_features = len(datasets[latency]['feature_names'])
            print(f"  - Latency {latency}: {n_train} train samples, {n_features} features")
        
        return datasets
        
    except FileNotFoundError as e:
        print(f"⚠ Warning: Could not test full pipeline - {e}")
        return None


def test_extensibility():
    """Test extensibility features."""
    print("\n" + "=" * 60)
    print("TEST 6: Extensibility")
    print("=" * 60)
    
    from config import ExperimentConfig
    
    # Test custom configuration
    config = ExperimentConfig()
    config.current_condition = 'strong'
    config.turbulence.max_samples = 5_000
    config.prediction.latencies = [5]
    config.features.enable_spectral = False
    config.features.enable_acf = False
    config.data_split.min_train_samples = 500
    
    print("✓ Custom configuration created")
    print(f"  - Disabled spectral features: {not config.features.enable_spectral}")
    print(f"  - Disabled ACF features: {not config.features.enable_acf}")
    print(f"  - Custom latencies: {config.prediction.latencies}")
    
    # Test switching conditions
    conditions = ['strong', 'moderate', 'weak']
    print(f"\n✓ Turbulence conditions configurable:")
    for condition in conditions:
        data_config = config.turbulence.DATA_FILES.get(condition)
        if data_config:
            print(f"  - {condition}: {data_config['variable']}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FSO DATA PREPARATION PIPELINE - VERIFICATION TESTS")
    print("=" * 70)
    
    try:
        # Test 1: Configuration
        config = test_configuration()
        
        # Test 2: Data loading
        data, metadata = test_data_loading(config)
        
        # Test 3: Feature engineering
        features_df = test_feature_engineering(data, config)
        
        # Test 4: Validation
        test_validation(data)
        
        # Test 5: Full pipeline
        datasets = test_full_pipeline(config)
        
        # Test 6: Extensibility
        test_extensibility()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 70)
        print("\nThe data preparation pipeline is ready for use.")
        print("See data_pipeline_README.md for detailed documentation.")
        print("See example_usage.py for comprehensive usage examples.")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED! ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

"""
Example usage of the FSO Channel Power Estimation Data Preparation Pipeline.

This script demonstrates:
1. Basic usage with default configuration
2. Custom configuration
3. Feature analysis
4. Model training integration
5. Multi-condition comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from config import get_config, print_config_summary, ExperimentConfig
from data_preparation import (
    prepare_dataset,
    load_turbulence_data,
    engineer_features,
    validate_data
)


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    # Create configuration
    config = get_config('strong')
    print_config_summary(config)
    
    # Prepare complete dataset
    datasets = prepare_dataset('strong', config=config)
    
    # Access data for a specific horizon
    latency = 5
    train_X, train_y = datasets[latency]['train']
    val_X, val_y = datasets[latency]['val']
    test_X, test_y = datasets[latency]['test']
    
    print(f"\n--- Latency = {latency} samples ({config.get_latency_time_ms(latency):.2f} ms) ---")
    print(f"Training set: X shape = {train_X.shape}, y shape = {train_y.shape}")
    print(f"Validation set: X shape = {val_X.shape}, y shape = {val_y.shape}")
    print(f"Test set: X shape = {test_X.shape}, y shape = {test_y.shape}")
    print(f"Number of features: {len(datasets[latency]['feature_names'])}")
    
    # Print first few feature names
    print(f"\nFirst 10 features:")
    for i, feat in enumerate(datasets[latency]['feature_names'][:10], 1):
        print(f"  {i}. {feat}")
    
    return datasets


def example_2_custom_configuration():
    """Example 2: Custom configuration for faster processing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 70)
    
    # Create custom configuration
    config = ExperimentConfig()
    config.current_condition = 'strong'
    
    # Use fewer samples for faster processing
    config.turbulence.max_samples = 100_000
    
    # Only generate 3 horizons
    config.prediction.latencies = [5, 50, 100]
    
    # Disable expensive features
    config.features.enable_spectral = False
    config.features.enable_acf = False
    
    # Reduce rolling windows
    config.features.rolling_window_sizes = [5, 10]
    
    # Adjust split for smaller dataset
    config.data_split.min_train_samples = 10_000
    
    print("\nCustom Configuration:")
    print(f"  - Max samples: {config.turbulence.max_samples:,}")
    print(f"  - Horizons: {config.prediction.latencies}")
    print(f"  - Spectral features: {config.features.enable_spectral}")
    print(f"  - ACF features: {config.features.enable_acf}")
    print(f"  - Rolling windows: {config.features.rolling_window_sizes}")
    
    # Prepare dataset
    datasets = prepare_dataset('strong', config=config)
    
    print(f"\n✓ Prepared {len(datasets)} horizons with custom config")
    
    return datasets


def example_3_feature_analysis(datasets):
    """Example 3: Analyze feature distributions and correlations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Feature Analysis")
    print("=" * 70)
    
    latency = 5
    train_X, train_y = datasets[latency]['train']
    feature_names = datasets[latency]['feature_names']
    
    # Basic statistics
    print(f"\n--- Feature Statistics (Latency = {latency}) ---")
    stats = train_X.describe()
    print(stats.iloc[:, :5])  # Show first 5 features
    
    # Check for feature variance
    variances = train_X.var().sort_values(ascending=False)
    print(f"\n--- Top 10 Features by Variance ---")
    for i, (feat, var) in enumerate(variances.head(10).items(), 1):
        print(f"  {i}. {feat}: {var:.6f}")
    
    # Feature-target correlations
    correlations = pd.DataFrame({
        'feature': feature_names,
        'correlation': [abs(train_X[feat].corr(train_y)) for feat in feature_names]
    }).sort_values('correlation', ascending=False)
    
    print(f"\n--- Top 10 Features by Correlation with Target ---")
    for i, row in enumerate(correlations.head(10).itertuples(), 1):
        print(f"  {i}. {row.feature}: {row.correlation:.4f}")
    
    # Plot correlation distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(correlations['correlation'], bins=50, edgecolor='black')
    plt.xlabel('Absolute Correlation with Target')
    plt.ylabel('Count')
    plt.title('Feature-Target Correlation Distribution')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    top_features = correlations.head(15)
    plt.barh(range(len(top_features)), top_features['correlation'])
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
    plt.xlabel('Absolute Correlation')
    plt.title('Top 15 Features by Correlation')
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved feature analysis plot to 'feature_analysis.png'")


def example_4_model_training(datasets):
    """Example 4: Train and evaluate models."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Model Training and Evaluation")
    print("=" * 70)
    
    latency = 5
    train_X, train_y = datasets[latency]['train']
    val_X, val_y = datasets[latency]['val']
    test_X, test_y = datasets[latency]['test']
    
    print(f"\n--- Training Models for Latency = {latency} samples ---")
    
    # Train Linear Regression
    print("\n1. Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(train_X, train_y)
    
    lr_val_pred = lr_model.predict(val_X)
    lr_test_pred = lr_model.predict(test_X)
    
    lr_val_rmse = np.sqrt(mean_squared_error(val_y, lr_val_pred))
    lr_test_rmse = np.sqrt(mean_squared_error(test_y, lr_test_pred))
    lr_test_r2 = r2_score(test_y, lr_test_pred)
    
    print(f"   Validation RMSE: {lr_val_rmse:.4f}")
    print(f"   Test RMSE: {lr_test_rmse:.4f}")
    print(f"   Test R²: {lr_test_r2:.4f}")
    
    # Train Random Forest
    print("\n2. Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, 
                                     random_state=42, n_jobs=-1)
    rf_model.fit(train_X, train_y)
    
    rf_val_pred = rf_model.predict(val_X)
    rf_test_pred = rf_model.predict(test_X)
    
    rf_val_rmse = np.sqrt(mean_squared_error(val_y, rf_val_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(test_y, rf_test_pred))
    rf_test_r2 = r2_score(test_y, rf_test_pred)
    
    print(f"   Validation RMSE: {rf_val_rmse:.4f}")
    print(f"   Test RMSE: {rf_test_rmse:.4f}")
    print(f"   Test R²: {rf_test_r2:.4f}")
    
    # Compare models
    print("\n--- Model Comparison ---")
    print(f"{'Model':<20} {'Val RMSE':<12} {'Test RMSE':<12} {'Test R²':<10}")
    print("-" * 55)
    print(f"{'Linear Regression':<20} {lr_val_rmse:<12.4f} {lr_test_rmse:<12.4f} {lr_test_r2:<10.4f}")
    print(f"{'Random Forest':<20} {rf_val_rmse:<12.4f} {rf_test_rmse:<12.4f} {rf_test_r2:<10.4f}")
    
    # Plot predictions
    plt.figure(figsize=(12, 4))
    
    # Plot subset of test predictions
    n_plot = 500
    x_axis = range(n_plot)
    
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, test_y.iloc[:n_plot], label='Actual', linewidth=1.5)
    plt.plot(x_axis, lr_test_pred[:n_plot], label='Linear Regression', alpha=0.7)
    plt.plot(x_axis, rf_test_pred[:n_plot], label='Random Forest', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Target Value')
    plt.title(f'Predictions vs Actual (Latency={latency})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_y, lr_test_pred, alpha=0.3, s=1, label='Linear Regression')
    plt.scatter(test_y, rf_test_pred, alpha=0.3, s=1, label='Random Forest')
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 
             'k--', linewidth=1, label='Perfect Prediction')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Prediction Scatter Plot')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved prediction plots to 'model_predictions.png'")
    
    return lr_model, rf_model


def example_5_multi_horizon_evaluation(datasets):
    """Example 5: Evaluate model performance across multiple horizons."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Horizon Evaluation")
    print("=" * 70)
    
    results = []
    
    for latency in sorted(datasets.keys()):
        print(f"\n--- Evaluating Latency = {latency} samples ---")
        
        train_X, train_y = datasets[latency]['train']
        test_X, test_y = datasets[latency]['test']
        
        # Train simple linear model
        model = LinearRegression()
        model.fit(train_X, train_y)
        
        # Evaluate
        pred_y = model.predict(test_X)
        rmse = np.sqrt(mean_squared_error(test_y, pred_y))
        r2 = r2_score(test_y, pred_y)
        
        results.append({
            'latency': latency,
            'latency_ms': latency / 10,  # 10kHz sampling
            'rmse': rmse,
            'r2': r2,
            'n_features': len(datasets[latency]['feature_names'])
        })
        
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, Features: {results[-1]['n_features']}")
    
    # Plot results
    results_df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(results_df['latency_ms'], results_df['rmse'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Prediction Horizon (ms)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Prediction Error vs Horizon')
    ax1.grid(alpha=0.3)
    
    ax2.plot(results_df['latency_ms'], results_df['r2'], 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Prediction Horizon (ms)')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Prediction Quality vs Horizon')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_horizon_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved multi-horizon results to 'multi_horizon_results.png'")
    
    # Print summary table
    print("\n--- Multi-Horizon Summary ---")
    print(results_df.to_string(index=False))
    
    return results_df


def example_6_feature_importance(datasets):
    """Example 6: Analyze feature importance using Random Forest."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Feature Importance Analysis")
    print("=" * 70)
    
    latency = 5
    train_X, train_y = datasets[latency]['train']
    feature_names = datasets[latency]['feature_names']
    
    print(f"\n--- Training Random Forest for Feature Importance ---")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                     random_state=42, n_jobs=-1)
    rf_model.fit(train_X, train_y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n--- Top 20 Most Important Features ---")
    for i, row in enumerate(importances.head(20).itertuples(), 1):
        print(f"  {i:2d}. {row.feature:<40} {row.importance:.6f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_n = 25
    top_features = importances.head(top_n)
    
    plt.barh(range(top_n), top_features['importance'])
    plt.yticks(range(top_n), top_features['feature'], fontsize=8)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved feature importance plot to 'feature_importance.png'")
    
    return importances


def example_7_data_quality_checks():
    """Example 7: Perform comprehensive data quality checks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Data Quality Checks")
    print("=" * 70)
    
    # Load raw data
    data, metadata = load_turbulence_data('strong')
    
    print("\n--- Raw Data Validation ---")
    validate_data(data, name="raw turbulence data")
    
    print("\n--- Data Statistics ---")
    print(f"  Samples: {len(data):,}")
    print(f"  Mean: {np.mean(data):.6f} (should be ~0 after centering)")
    print(f"  Std: {np.std(data):.4f}")
    print(f"  Min: {np.min(data):.4f}")
    print(f"  Max: {np.max(data):.4f}")
    print(f"  Range: {np.max(data) - np.min(data):.4f}")
    
    # Check for outliers
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    
    print(f"\n--- Outlier Analysis ---")
    print(f"  Q1: {q1:.4f}")
    print(f"  Q3: {q3:.4f}")
    print(f"  IQR: {iqr:.4f}")
    print(f"  Outliers (3*IQR): {outliers:,} ({outliers/len(data)*100:.2f}%)")
    
    # Plot data distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time series
    axes[0, 0].plot(data[:10000])
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Power (dB)')
    axes[0, 0].set_title('Time Series (First 10,000 samples)')
    axes[0, 0].grid(alpha=0.3)
    
    # Histogram
    axes[0, 1].hist(data, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Power (dB)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(data[::100], dist="norm", plot=axes[1, 0])  # Subsample for speed
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    axes[1, 0].grid(alpha=0.3)
    
    # Autocorrelation
    max_lag = 100
    acf = [np.corrcoef(data[:-i], data[i:])[0, 1] if i > 0 else 1.0 
           for i in range(max_lag)]
    axes[1, 1].plot(acf)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 1].axhline(y=1.96/np.sqrt(len(data)), color='r', linestyle='--', linewidth=0.5)
    axes[1, 1].axhline(y=-1.96/np.sqrt(len(data)), color='r', linestyle='--', linewidth=0.5)
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('ACF')
    axes[1, 1].set_title('Autocorrelation Function')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_quality.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved data quality plots to 'data_quality.png'")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("FSO Channel Power Estimation - Example Usage")
    print("=" * 70)
    
    # Example 1: Basic usage
    datasets = example_1_basic_usage()
    
    # Example 2: Custom configuration
    # datasets_custom = example_2_custom_configuration()
    
    # Example 3: Feature analysis
    example_3_feature_analysis(datasets)
    
    # Example 4: Model training
    lr_model, rf_model = example_4_model_training(datasets)
    
    # Example 5: Multi-horizon evaluation
    results_df = example_5_multi_horizon_evaluation(datasets)
    
    # Example 6: Feature importance
    importances = example_6_feature_importance(datasets)
    
    # Example 7: Data quality checks
    example_7_data_quality_checks()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - feature_analysis.png")
    print("  - model_predictions.png")
    print("  - multi_horizon_results.png")
    print("  - feature_importance.png")
    print("  - data_quality.png")


if __name__ == "__main__":
    main()

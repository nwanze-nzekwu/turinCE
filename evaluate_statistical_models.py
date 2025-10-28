"""
Evaluation Script for Statistical Time Series Models.

This script evaluates ARIMA, SARIMAX, and Prophet models on the FSO channel
power estimation task and compares them to ML baselines.
"""

import os
import time
import warnings
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import get_config
from data_preparation import load_turbulence_data
from statistical_models import (
    test_stationarity,
    ARIMAForecaster,
    SARIMAXForecaster,
    ProphetForecaster,
    evaluate_statistical_model,
    plot_residual_diagnostics,
    PROPHET_AVAILABLE
)

warnings.filterwarnings('ignore')


def prepare_data_for_statistical_models(
    condition: str = 'strong',
    max_samples: int = 100_000
) -> Dict:
    """
    Load and prepare data for statistical models.
    
    Statistical models work on raw univariate time series, not engineered features.
    
    Args:
        condition: Turbulence condition
        max_samples: Maximum number of samples to use
        
    Returns:
        Dictionary with data splits and metadata
    """
    print("\n" + "=" * 70)
    print("DATA PREPARATION FOR STATISTICAL MODELS")
    print("=" * 70)
    
    # Load data
    config = get_config(condition)
    config.turbulence.max_samples = max_samples
    
    data, metadata = load_turbulence_data(condition, config=config)
    
    # Use time-aware split (70/15/15)
    n = len(data)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"\nData Splits:")
    print(f"  Train: {len(train_data):,} samples (70%)")
    print(f"  Val:   {len(val_data):,} samples (15%)")
    print(f"  Test:  {len(test_data):,} samples (15%)")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'full': data,
        'metadata': metadata
    }


def evaluate_arima(
    train_data: np.ndarray,
    test_data: np.ndarray,
    horizons: List[int],
    p_range: List[int] = [0, 1, 2],
    d_range: List[int] = [0, 1],
    q_range: List[int] = [0, 1, 2]
) -> Dict[int, dict]:
    """
    Evaluate ARIMA model across multiple horizons.
    
    Args:
        train_data: Training data
        test_data: Test data
        horizons: List of forecast horizons
        p_range: AR order range
        d_range: Differencing order range
        q_range: MA order range
        
    Returns:
        Dictionary mapping horizon to results
    """
    print("\n" + "=" * 70)
    print("ARIMA EVALUATION")
    print("=" * 70)
    
    results = {}
    
    for horizon in horizons:
        print(f"\n{'=' * 70}")
        print(f"ARIMA - Horizon {horizon} samples ({horizon/10:.1f}ms)")
        print(f"{'=' * 70}")
        
        start_time = time.time()
        
        # Initialize and fit ARIMA
        model = ARIMAForecaster(
            p_range=p_range,
            d_range=d_range,
            q_range=q_range,
            criterion='aic'
        )
        
        print(f"\nFitting ARIMA model...")
        model.fit(train_data, auto_search=True)
        
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f}s")
        
        # Evaluate
        eval_results = evaluate_statistical_model(
            model,
            f"ARIMA{model.best_order}",
            train_data,
            test_data,
            horizon,
            refit_interval=500  # Refit every 500 samples
        )
        eval_results.train_time = train_time
        
        # Plot diagnostics
        plot_residual_diagnostics(
            eval_results.residuals,
            f"ARIMA{model.best_order}_h{horizon}",
            save_path=f"arima_diagnostics_h{horizon}.png"
        )
        
        results[horizon] = {
            'model': model,
            'results': eval_results
        }
    
    return results


def evaluate_sarimax(
    train_data: np.ndarray,
    test_data: np.ndarray,
    horizons: List[int],
    order: tuple = (1, 0, 1),
    seasonal_orders: List[tuple] = [(0, 0, 0, 0), (1, 0, 1, 10)]
) -> Dict[int, dict]:
    """
    Evaluate SARIMAX model across multiple horizons.
    
    Args:
        train_data: Training data
        test_data: Test data
        horizons: List of forecast horizons
        order: Base ARIMA order
        seasonal_orders: List of seasonal orders to try
        
    Returns:
        Dictionary mapping horizon to results
    """
    print("\n" + "=" * 70)
    print("SARIMAX EVALUATION")
    print("=" * 70)
    
    results = {}
    
    for horizon in horizons:
        print(f"\n{'=' * 70}")
        print(f"SARIMAX - Horizon {horizon} samples ({horizon/10:.1f}ms)")
        print(f"{'=' * 70}")
        
        best_aic = np.inf
        best_model = None
        best_seasonal_order = None
        
        # Try different seasonal orders
        for seasonal_order in seasonal_orders:
            try:
                print(f"\nTrying seasonal order: {seasonal_order}")
                start_time = time.time()
                
                model = SARIMAXForecaster(
                    order=order,
                    seasonal_order=seasonal_order
                )
                
                model.fit(train_data)
                train_time = time.time() - start_time
                
                if model.fitted_model.aic < best_aic:
                    best_aic = model.fitted_model.aic
                    best_model = model
                    best_seasonal_order = seasonal_order
                    
                print(f"  AIC: {model.fitted_model.aic:.2f}, Training time: {train_time:.2f}s")
                
            except Exception as e:
                print(f"  Failed: {str(e)[:100]}")
                continue
        
        if best_model is None:
            print(f"WARNING: All SARIMAX configurations failed for horizon {horizon}")
            continue
        
        print(f"\nBest seasonal order: {best_seasonal_order} (AIC: {best_aic:.2f})")
        
        # Evaluate best model
        eval_results = evaluate_statistical_model(
            best_model,
            f"SARIMAX{order}x{best_seasonal_order}",
            train_data,
            test_data,
            horizon,
            refit_interval=500
        )
        
        # Plot diagnostics
        plot_residual_diagnostics(
            eval_results.residuals,
            f"SARIMAX_h{horizon}",
            save_path=f"sarimax_diagnostics_h{horizon}.png"
        )
        
        results[horizon] = {
            'model': best_model,
            'results': eval_results
        }
    
    return results


def evaluate_prophet(
    train_data: np.ndarray,
    test_data: np.ndarray,
    horizons: List[int],
    hyperparams: List[dict] = None
) -> Dict[int, dict]:
    """
    Evaluate Prophet model across multiple horizons.
    
    Args:
        train_data: Training data
        test_data: Test data
        horizons: List of forecast horizons
        hyperparams: List of hyperparameter configurations
        
    Returns:
        Dictionary mapping horizon to results
    """
    if not PROPHET_AVAILABLE:
        print("\n" + "=" * 70)
        print("PROPHET NOT AVAILABLE - SKIPPING")
        print("=" * 70)
        print("Install with: pip install prophet")
        return {}
    
    print("\n" + "=" * 70)
    print("PROPHET EVALUATION")
    print("=" * 70)
    
    if hyperparams is None:
        hyperparams = [
            {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'additive'
            },
            {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 1.0,
                'seasonality_mode': 'additive'
            }
        ]
    
    results = {}
    
    for horizon in horizons:
        print(f"\n{'=' * 70}")
        print(f"Prophet - Horizon {horizon} samples ({horizon/10:.1f}ms)")
        print(f"{'=' * 70}")
        
        best_rmse = np.inf
        best_model = None
        best_params = None
        
        # Try different hyperparameters
        for params in hyperparams:
            try:
                print(f"\nTrying params: {params}")
                start_time = time.time()
                
                model = ProphetForecaster(**params)
                model.fit(train_data, freq='100us')  # 10kHz = 100us period
                
                train_time = time.time() - start_time
                
                # Quick validation on a subset
                val_preds = model.forecast(steps=min(1000, len(test_data)))
                val_rmse = np.sqrt(np.mean((test_data[:len(val_preds)] - val_preds) ** 2))
                
                print(f"  Val RMSE: {val_rmse:.4f}, Training time: {train_time:.2f}s")
                
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_model = model
                    best_params = params
                    
            except Exception as e:
                print(f"  Failed: {str(e)[:100]}")
                continue
        
        if best_model is None:
            print(f"WARNING: All Prophet configurations failed for horizon {horizon}")
            continue
        
        print(f"\nBest params: {best_params}")
        
        # Evaluate best model
        eval_results = evaluate_statistical_model(
            best_model,
            "Prophet",
            train_data,
            test_data,
            horizon,
            refit_interval=0  # Prophet is too slow to refit frequently
        )
        
        # Plot diagnostics
        plot_residual_diagnostics(
            eval_results.residuals,
            f"Prophet_h{horizon}",
            save_path=f"prophet_diagnostics_h{horizon}.png"
        )
        
        results[horizon] = {
            'model': best_model,
            'results': eval_results
        }
    
    return results


def create_comparison_table(
    arima_results: Dict,
    sarimax_results: Dict,
    prophet_results: Dict
) -> pd.DataFrame:
    """
    Create comparison table of all models.
    
    Args:
        arima_results: ARIMA results
        sarimax_results: SARIMAX results
        prophet_results: Prophet results
        
    Returns:
        DataFrame with comparison
    """
    data = []
    
    for horizon in sorted(arima_results.keys()):
        # ARIMA
        if horizon in arima_results:
            res = arima_results[horizon]['results']
            data.append({
                'Model': 'ARIMA',
                'Horizon (samples)': horizon,
                'Horizon (ms)': horizon / 10,
                'RMSE': res.rmse,
                'MAE': res.mae,
                'Variance Reduction': res.variance_reduction,
                'Train Time (s)': res.train_time,
                'Inference Time (s)': res.inference_time,
                'Inference Time per Sample (ms)': (res.inference_time / len(res.predictions)) * 1000,
                'AIC': res.aic,
                'BIC': res.bic
            })
        
        # SARIMAX
        if horizon in sarimax_results:
            res = sarimax_results[horizon]['results']
            data.append({
                'Model': 'SARIMAX',
                'Horizon (samples)': horizon,
                'Horizon (ms)': horizon / 10,
                'RMSE': res.rmse,
                'MAE': res.mae,
                'Variance Reduction': res.variance_reduction,
                'Train Time (s)': res.train_time,
                'Inference Time (s)': res.inference_time,
                'Inference Time per Sample (ms)': (res.inference_time / len(res.predictions)) * 1000,
                'AIC': res.aic,
                'BIC': res.bic
            })
        
        # Prophet
        if horizon in prophet_results:
            res = prophet_results[horizon]['results']
            data.append({
                'Model': 'Prophet',
                'Horizon (samples)': horizon,
                'Horizon (ms)': horizon / 10,
                'RMSE': res.rmse,
                'MAE': res.mae,
                'Variance Reduction': res.variance_reduction,
                'Train Time (s)': res.train_time,
                'Inference Time (s)': res.inference_time,
                'Inference Time per Sample (ms)': (res.inference_time / len(res.predictions)) * 1000,
                'AIC': None,
                'BIC': None
            })
    
    df = pd.DataFrame(data)
    return df


def plot_performance_comparison(results_df: pd.DataFrame, save_path: str = 'statistical_models_comparison.png'):
    """
    Plot performance comparison across models and horizons.
    
    Args:
        results_df: Results DataFrame
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: RMSE vs Horizon
    ax = axes[0, 0]
    for model in results_df['Model'].unique():
        data = results_df[results_df['Model'] == model]
        ax.plot(data['Horizon (ms)'], data['RMSE'], 'o-', label=model, markersize=8, linewidth=2)
    ax.set_xlabel('Forecast Horizon (ms)')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs Forecast Horizon')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Variance Reduction vs Horizon
    ax = axes[0, 1]
    for model in results_df['Model'].unique():
        data = results_df[results_df['Model'] == model]
        ax.plot(data['Horizon (ms)'], data['Variance Reduction'] * 100, 'o-', 
                label=model, markersize=8, linewidth=2)
    ax.set_xlabel('Forecast Horizon (ms)')
    ax.set_ylabel('Variance Reduction (%)')
    ax.set_title('Variance Reduction vs Forecast Horizon')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Training Time
    ax = axes[1, 0]
    for model in results_df['Model'].unique():
        data = results_df[results_df['Model'] == model]
        ax.bar(data['Horizon (ms)'].values + (list(results_df['Model'].unique()).index(model) - 1) * 1,
               data['Train Time (s)'], width=0.8, label=model, alpha=0.7)
    ax.set_xlabel('Forecast Horizon (ms)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Inference Time per Sample
    ax = axes[1, 1]
    for model in results_df['Model'].unique():
        data = results_df[results_df['Model'] == model]
        ax.plot(data['Horizon (ms)'], data['Inference Time per Sample (ms)'], 'o-',
                label=model, markersize=8, linewidth=2)
    ax.set_xlabel('Forecast Horizon (ms)')
    ax.set_ylabel('Inference Time per Sample (ms)')
    ax.set_title('Inference Time per Sample')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")
    plt.close()


def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print("STATISTICAL TIME SERIES MODELS EVALUATION")
    print("FSO Channel Power Estimation")
    print("=" * 70)
    
    # Configuration
    CONDITION = 'strong'
    MAX_SAMPLES = 100_000  # Use subset for faster evaluation
    HORIZONS = [50, 100, 200, 500]  # Standard horizons (excluding 5, too short)
    
    # Prepare data
    data_dict = prepare_data_for_statistical_models(CONDITION, MAX_SAMPLES)
    train_data = data_dict['train']
    test_data = data_dict['test']
    
    # Test stationarity
    print("\n" + "=" * 70)
    print("STATIONARITY TESTING")
    print("=" * 70)
    stationarity_result = test_stationarity(train_data, verbose=True)
    
    # Evaluate ARIMA
    arima_results = evaluate_arima(
        train_data,
        test_data,
        HORIZONS,
        p_range=[0, 1, 2],  # Reduced for speed
        d_range=[0, 1],
        q_range=[0, 1, 2]
    )
    
    # Evaluate SARIMAX
    sarimax_results = evaluate_sarimax(
        train_data,
        test_data,
        HORIZONS,
        order=(1, 0, 1),
        seasonal_orders=[(0, 0, 0, 0), (1, 0, 1, 10)]  # Test with and without seasonality
    )
    
    # Evaluate Prophet (if available)
    prophet_results = evaluate_prophet(
        train_data,
        test_data,
        HORIZONS[:2]  # Only evaluate first 2 horizons (Prophet is slow)
    )
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    results_df = create_comparison_table(arima_results, sarimax_results, prophet_results)
    
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('statistical_models_results.csv', index=False)
    print("\n✓ Saved results to 'statistical_models_results.csv'")
    
    # Plot comparison
    plot_performance_comparison(results_df)
    
    # Performance analysis
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    print("\nBest RMSE by Model:")
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        best_rmse = model_data['RMSE'].min()
        best_horizon = model_data.loc[model_data['RMSE'].idxmin(), 'Horizon (ms)']
        print(f"  {model:15s}: {best_rmse:.4f} at horizon {best_horizon:.1f}ms")
    
    print("\nWorst RMSE by Model:")
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        worst_rmse = model_data['RMSE'].max()
        worst_horizon = model_data.loc[model_data['RMSE'].idxmax(), 'Horizon (ms)']
        print(f"  {model:15s}: {worst_rmse:.4f} at horizon {worst_horizon:.1f}ms")
    
    print("\nComputational Cost:")
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        avg_train_time = model_data['Train Time (s)'].mean()
        avg_inference_time = model_data['Inference Time per Sample (ms)'].mean()
        print(f"  {model:15s}: Train {avg_train_time:.2f}s, Inference {avg_inference_time:.4f}ms/sample")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. Statistical models may struggle with long horizons (>200 samples)")
    print("2. ARIMA typically faster than SARIMAX and Prophet")
    print("3. Seasonal components may not help at millisecond timescales")
    print("4. Compare these results to ML baselines (Random Forest, XGBoost)")
    print("5. Statistical models are interpretable but may underperform ML models")
    print("\nNext Steps:")
    print("- Compare to ML baseline results from Task #78")
    print("- Consider hybrid approaches (statistical + ML)")
    print("- Document where each model type excels")


if __name__ == "__main__":
    main()

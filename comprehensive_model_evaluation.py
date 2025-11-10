"""
Comprehensive Model Evaluation for FSO Channel Power Estimation.

This script evaluates ALL implemented models across multiple prediction horizons:
- Baseline models: Naive, Linear Regression
- Tree-based models: Random Forest, XGBoost, LightGBM  
- Deep learning models: LSTM, GRU, Transformer

Generates comprehensive comparison tables, visualizations, and summary report.
"""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import get_config
from data_preparation import prepare_dataset, load_turbulence_data
from gradient_boosting_models import GradientBoostingTrainer, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE

# Try importing deep learning modules
try:
    from deep_learning_models import TORCH_AVAILABLE, DeepLearningForecaster, set_random_seeds
    DL_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    DL_AVAILABLE = False
    warnings.warn("Deep learning modules not available")

warnings.filterwarnings('ignore')


def evaluate_naive_baseline(
    datasets: Dict,
    horizon: int
) -> Dict:
    """
    Evaluate naive baseline (persistence model: prediction = last value).
    
    Args:
        datasets: Dictionary of datasets for the horizon
        horizon: Prediction horizon
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n--- Evaluating Naive Baseline for Horizon {horizon} ---")
    
    # For naive model, prediction is just the last known value
    # In our feature engineering, this corresponds to lag_5 (or lag at latency)
    # Since we're predicting the difference, naive prediction = 0 (no change)
    
    train_X, train_y = datasets[horizon]['train']
    val_X, val_y = datasets[horizon]['val']
    test_X, test_y = datasets[horizon]['test']
    
    # Naive prediction: assume no change (predict 0 for the difference)
    start_time = time.time()
    train_pred = np.zeros(len(train_y))
    val_pred = np.zeros(len(val_y))
    test_pred = np.zeros(len(test_y))
    inference_time = time.time() - start_time
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
    val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    test_mae = mean_absolute_error(test_y, test_pred)
    test_r2 = r2_score(test_y, test_pred)
    
    # Residual variance
    residuals = test_y - test_pred
    residual_variance = np.var(residuals)
    
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Val RMSE: {val_rmse:.6f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    print(f"  Test R²: {test_r2:.6f}")
    
    return {
        'model_name': 'Naive',
        'horizon': horizon,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'residual_variance': residual_variance,
        'training_time': 0.0,  # No training required
        'inference_time_per_sample_ms': (inference_time / len(test_X)) * 1000 if len(test_X) > 0 else 0.0,
        'model_complexity': 0,  # No parameters
        'n_train_samples': len(train_X),
        'n_test_samples': len(test_X)
    }


def evaluate_linear_regression(
    datasets: Dict,
    horizon: int,
    random_state: int = 42
) -> Dict:
    """
    Train and evaluate Linear Regression model.
    
    Args:
        datasets: Dictionary of datasets for the horizon
        horizon: Prediction horizon
        random_state: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n--- Training and Evaluating Linear Regression for Horizon {horizon} ---")
    
    train_X, train_y = datasets[horizon]['train']
    val_X, val_y = datasets[horizon]['val']
    test_X, test_y = datasets[horizon]['test']
    
    # Train model
    start_time = time.time()
    model = LinearRegression()
    model.fit(train_X, train_y)
    training_time = time.time() - start_time
    
    # Predictions
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    
    start_time = time.time()
    test_pred = model.predict(test_X)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
    val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    test_mae = mean_absolute_error(test_y, test_pred)
    test_r2 = r2_score(test_y, test_pred)
    
    # Residual variance
    residuals = test_y - test_pred
    residual_variance = np.var(residuals)
    
    # Model complexity
    n_features = train_X.shape[1]
    n_params = n_features + 1  # coefficients + intercept
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Val RMSE: {val_rmse:.6f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    print(f"  Test R²: {test_r2:.6f}")
    
    return {
        'model_name': 'Linear Regression',
        'horizon': horizon,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'residual_variance': residual_variance,
        'training_time': training_time,
        'inference_time_per_sample_ms': (inference_time / len(test_X)) * 1000,
        'model_complexity': n_params,
        'n_train_samples': len(train_X),
        'n_test_samples': len(test_X)
    }


def evaluate_random_forest(
    datasets: Dict,
    horizon: int,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10
) -> Dict:
    """
    Train and evaluate Random Forest model.
    
    Args:
        datasets: Dictionary of datasets for the horizon
        horizon: Prediction horizon
        random_state: Random seed
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n--- Training and Evaluating Random Forest for Horizon {horizon} ---")
    
    train_X, train_y = datasets[horizon]['train']
    val_X, val_y = datasets[horizon]['val']
    test_X, test_y = datasets[horizon]['test']
    
    # Train model
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(train_X, train_y)
    training_time = time.time() - start_time
    
    # Predictions
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    
    start_time = time.time()
    test_pred = model.predict(test_X)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
    val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    test_mae = mean_absolute_error(test_y, test_pred)
    test_r2 = r2_score(test_y, test_pred)
    
    # Residual variance
    residuals = test_y - test_pred
    residual_variance = np.var(residuals)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Val RMSE: {val_rmse:.6f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    print(f"  Test R²: {test_r2:.6f}")
    
    return {
        'model_name': 'Random Forest',
        'horizon': horizon,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'residual_variance': residual_variance,
        'training_time': training_time,
        'inference_time_per_sample_ms': (inference_time / len(test_X)) * 1000,
        'model_complexity': n_estimators,  # Number of trees
        'n_train_samples': len(train_X),
        'n_test_samples': len(test_X)
    }


def evaluate_gradient_boosting(
    datasets: Dict,
    horizon: int,
    model_type: str,
    random_state: int = 42,
    tune_params: bool = False,
    max_tuning_trials: int = 30,
    use_gpu: bool = False
) -> Dict:
    """
    Train and evaluate gradient boosting models (XGBoost or LightGBM).
    
    Args:
        datasets: Dictionary of datasets for the horizon
        horizon: Prediction horizon
        model_type: 'xgboost' or 'lightgbm'
        random_state: Random seed
        tune_params: Whether to perform hyperparameter tuning
        max_tuning_trials: Maximum tuning trials
        use_gpu: Whether to use GPU (XGBoost only)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n--- Training and Evaluating {model_type.upper()} for Horizon {horizon} ---")
    
    train_X, train_y = datasets[horizon]['train']
    val_X, val_y = datasets[horizon]['val']
    test_X, test_y = datasets[horizon]['test']
    
    trainer = GradientBoostingTrainer(random_state=random_state)
    
    # Hyperparameter tuning if requested
    best_params = None
    if tune_params:
        print(f"  Performing hyperparameter tuning ({max_tuning_trials} trials)...")
        best_params, _ = trainer.tune_hyperparameters(
            model_type,
            train_X, train_y,
            val_X, val_y,
            max_trials=max_tuning_trials,
            use_gpu=use_gpu
        )
        print(f"  Best parameters: {best_params}")
    
    # Train final model
    if model_type == 'xgboost':
        model, train_metrics = trainer.train_xgboost(
            train_X, train_y, val_X, val_y,
            params=best_params,
            use_gpu=use_gpu
        )
    elif model_type == 'lightgbm':
        model, train_metrics = trainer.train_lightgbm(
            train_X, train_y, val_X, val_y,
            params=best_params
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate on test set
    start_time = time.time()
    test_pred = model.predict(test_X)
    inference_time = time.time() - start_time
    
    test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    test_mae = mean_absolute_error(test_y, test_pred)
    test_r2 = r2_score(test_y, test_pred)
    
    # Residual variance
    residuals = test_y - test_pred
    residual_variance = np.var(residuals)
    
    print(f"  Training time: {train_metrics['training_time']:.2f}s")
    print(f"  Train RMSE: {train_metrics['train_rmse']:.6f}")
    print(f"  Val RMSE: {train_metrics['val_rmse']:.6f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    print(f"  Test R²: {test_r2:.6f}")
    
    # Get model complexity
    if model_type == 'xgboost':
        n_estimators = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    else:  # lightgbm
        n_estimators = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
    
    return {
        'model_name': 'XGBoost' if model_type == 'xgboost' else 'LightGBM',
        'horizon': horizon,
        'train_rmse': train_metrics['train_rmse'],
        'val_rmse': train_metrics['val_rmse'],
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'residual_variance': residual_variance,
        'training_time': train_metrics['training_time'],
        'inference_time_per_sample_ms': (inference_time / len(test_X)) * 1000,
        'model_complexity': n_estimators,
        'n_train_samples': len(train_X),
        'n_test_samples': len(test_X)
    }


def evaluate_deep_learning(
    data: np.ndarray,
    horizon: int,
    model_type: str,
    lookback: int = None,
    random_state: int = 42,
    tune_params: bool = False,
    n_trials: int = 20,
    use_gpu: bool = True
) -> Dict:
    """
    Train and evaluate deep learning models (LSTM, GRU, Transformer).
    
    Args:
        data: Time series data
        horizon: Prediction horizon
        model_type: 'lstm', 'gru', or 'transformer'
        lookback: Number of past timesteps (default: 2*horizon, capped at 200)
        random_state: Random seed
        tune_params: Whether to perform hyperparameter tuning
        n_trials: Number of tuning trials
        use_gpu: Whether to use GPU
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n--- Training and Evaluating {model_type.upper()} for Horizon {horizon} ---")
    
    if not DL_AVAILABLE:
        print("  WARNING: Deep learning not available, skipping")
        return None
    
    # Set lookback
    if lookback is None:
        lookback = min(2 * horizon, 200)
    
    # Create forecaster
    forecaster = DeepLearningForecaster(
        model_type=model_type,
        lookback=lookback,
        horizon=horizon,
        random_seed=random_state,
        use_gpu=use_gpu
    )
    
    # Prepare data
    datasets = forecaster.prepare_data(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        normalization='standard'
    )
    
    train_X, train_y = datasets['train']
    val_X, val_y = datasets['val']
    test_X, test_y = datasets['test']
    
    # Hyperparameter tuning
    best_params = None
    if tune_params:
        print(f"  Performing hyperparameter tuning ({n_trials} trials)...")
        tuning_result = forecaster.tune(train_X, train_y, val_X, val_y, n_trials=n_trials)
        best_params = tuning_result['best_params']
        print(f"  Best parameters: {best_params}")
    
    # Train final model
    training_result = forecaster.train(
        train_X, train_y,
        val_X, val_y,
        params=best_params
    )
    
    # Evaluate on test set
    eval_result = forecaster.evaluate(test_X, test_y)
    
    # Calculate train RMSE for overfitting check
    train_eval = forecaster.evaluate(train_X, train_y)
    
    print(f"  Training time: {training_result['training_time']:.2f}s")
    print(f"  Train RMSE: {train_eval['test_rmse']:.6f}")
    print(f"  Val RMSE: {training_result['best_val_rmse']:.6f}")
    print(f"  Test RMSE: {eval_result['test_rmse']:.6f}")
    print(f"  Test MAE: {eval_result['test_mae']:.6f}")
    print(f"  Test R²: {eval_result['test_r2']:.6f}")
    
    # Residual variance
    test_pred = forecaster.predict(test_X)
    residuals = test_y.flatten() - test_pred.flatten()
    residual_variance = np.var(residuals)
    
    # Get model complexity (approximate parameter count)
    if hasattr(forecaster.model, 'count_parameters'):
        n_params = forecaster.model.count_parameters()
    else:
        n_params = sum(p.numel() for p in forecaster.model.parameters())
    
    return {
        'model_name': model_type.upper(),
        'horizon': horizon,
        'lookback': lookback,
        'train_rmse': train_eval['test_rmse'],
        'val_rmse': training_result['best_val_rmse'],
        'test_rmse': eval_result['test_rmse'],
        'test_mae': eval_result['test_mae'],
        'test_r2': eval_result['test_r2'],
        'residual_variance': residual_variance,
        'training_time': training_result['training_time'],
        'inference_time_per_sample_ms': eval_result['inference_time_per_sample_ms'],
        'model_complexity': n_params,
        'epochs_trained': training_result['epochs_trained'],
        'n_train_samples': len(train_X),
        'n_test_samples': len(test_X)
    }


def run_comprehensive_evaluation(
    condition: str = 'strong',
    horizons: List[int] = [50, 100, 200, 500],
    models: List[str] = ['naive', 'linear_regression', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru', 'transformer'],
    tune_gb_models: bool = False,
    tune_dl_models: bool = False,
    max_tuning_trials: int = 30,
    use_gpu: bool = False,
    random_state: int = 42,
    output_dir: str = 'results'
) -> pd.DataFrame:
    """
    Run comprehensive evaluation across all models and horizons.
    
    Args:
        condition: Turbulence condition
        horizons: List of prediction horizons to evaluate
        models: List of model types to evaluate
        tune_gb_models: Whether to tune gradient boosting models
        tune_dl_models: Whether to tune deep learning models
        max_tuning_trials: Maximum tuning trials
        use_gpu: Whether to use GPU
        random_state: Random seed
        output_dir: Output directory
        
    Returns:
        DataFrame with all results
    """
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION - FSO Channel Power Estimation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Turbulence condition: {condition}")
    print(f"  Prediction horizons: {horizons}")
    print(f"  Models to evaluate: {models}")
    print(f"  Tune gradient boosting: {tune_gb_models}")
    print(f"  Tune deep learning: {tune_dl_models}")
    print(f"  Max tuning trials: {max_tuning_trials}")
    print(f"  Use GPU: {use_gpu}")
    print(f"  Random state: {random_state}")
    print(f"  Output directory: {output_dir}")
    print("="*80)
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path = output_path / 'model_comparison_plots'
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare datasets for tabular models
    print("\n" + "="*80)
    print("STEP 1: Preparing Datasets")
    print("="*80)
    
    config = get_config(condition)
    config.prediction.latencies = horizons
    
    # For tabular models
    datasets = prepare_dataset(condition, config=config)
    print(f"\n✓ Datasets prepared for {len(datasets)} horizons")
    
    # For deep learning models
    dl_data = None
    if any(m in models for m in ['lstm', 'gru', 'transformer']):
        dl_data, metadata = load_turbulence_data(condition, config)
        print(f"✓ Loaded {len(dl_data):,} samples for deep learning")
    
    # Run evaluations
    print("\n" + "="*80)
    print("STEP 2: Evaluating Models")
    print("="*80)
    
    all_results = []
    total_start_time = time.time()
    
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"HORIZON: {horizon} samples ({horizon/10:.1f} ms @ 10kHz)")
        print(f"{'='*80}")
        
        # Baseline models
        if 'naive' in models:
            try:
                result = evaluate_naive_baseline(datasets, horizon)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR evaluating Naive: {e}")
        
        if 'linear_regression' in models:
            try:
                result = evaluate_linear_regression(datasets, horizon, random_state)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR evaluating Linear Regression: {e}")
        
        # Tree-based models
        if 'random_forest' in models:
            try:
                result = evaluate_random_forest(datasets, horizon, random_state)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR evaluating Random Forest: {e}")
        
        # Gradient boosting models
        if 'xgboost' in models and XGBOOST_AVAILABLE:
            try:
                result = evaluate_gradient_boosting(
                    datasets, horizon, 'xgboost',
                    random_state, tune_gb_models, max_tuning_trials, use_gpu
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR evaluating XGBoost: {e}")
        elif 'xgboost' in models:
            print("  WARNING: XGBoost not available, skipping")
        
        if 'lightgbm' in models and LIGHTGBM_AVAILABLE:
            try:
                result = evaluate_gradient_boosting(
                    datasets, horizon, 'lightgbm',
                    random_state, tune_gb_models, max_tuning_trials, use_gpu
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR evaluating LightGBM: {e}")
        elif 'lightgbm' in models:
            print("  WARNING: LightGBM not available, skipping")
        
        # Deep learning models
        if dl_data is not None:
            for dl_model in ['lstm', 'gru', 'transformer']:
                if dl_model in models:
                    try:
                        result = evaluate_deep_learning(
                            dl_data, horizon, dl_model,
                            random_state=random_state,
                            tune_params=tune_dl_models,
                            n_trials=max_tuning_trials,
                            use_gpu=use_gpu
                        )
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        print(f"ERROR evaluating {dl_model.upper()}: {e}")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE - Total time: {total_time/60:.1f} minutes")
    print("="*80)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_csv = output_path / 'comparison_table.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to {results_csv}")
    
    return results_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of all FSO power estimation models'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='strong',
        help='Turbulence condition (strong, moderate, weak)'
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[50, 100, 200, 500],
        help='Prediction horizons to evaluate'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['naive', 'linear_regression', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru', 'transformer'],
        choices=['naive', 'linear_regression', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru', 'transformer'],
        help='Models to evaluate'
    )
    parser.add_argument(
        '--tune-gb',
        action='store_true',
        help='Tune gradient boosting models'
    )
    parser.add_argument(
        '--tune-dl',
        action='store_true',
        help='Tune deep learning models'
    )
    parser.add_argument(
        '--max-tuning-trials',
        type=int,
        default=30,
        help='Maximum tuning trials'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results_df = run_comprehensive_evaluation(
        condition=args.condition,
        horizons=args.horizons,
        models=args.models,
        tune_gb_models=args.tune_gb,
        tune_dl_models=args.tune_dl,
        max_tuning_trials=args.max_tuning_trials,
        use_gpu=args.use_gpu,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.groupby('model_name')['test_rmse'].agg(['mean', 'min', 'max']))
    

if __name__ == '__main__':
    main()

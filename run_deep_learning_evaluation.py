"""
Complete Evaluation Script for Deep Learning Models.

This script trains and evaluates LSTM, GRU, and Transformer models
on multiple prediction horizons for FSO channel power estimation.
"""

import os
import argparse
import time
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

from data_preparation import load_turbulence_data
from config import get_config
from deep_learning_models import (
    TORCH_AVAILABLE,
    DeepLearningForecaster,
    set_random_seeds,
    get_device
)


def ensure_directories():
    """Create necessary directories for models and results."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("✓ Created models/ and results/ directories")


def train_and_evaluate_model(
    model_type: str,
    data: np.ndarray,
    lookback: int,
    horizon: int,
    tune_params: bool = False,
    n_trials: int = 20,
    random_seed: int = 42,
    use_gpu: bool = True,
    save_model: bool = True
) -> Dict:
    """
    Train and evaluate a single model configuration.
    
    Args:
        model_type: 'lstm', 'gru', or 'transformer'
        data: Time series data
        lookback: Number of past timesteps
        horizon: Prediction horizon
        tune_params: Whether to tune hyperparameters
        n_trials: Number of tuning trials
        random_seed: Random seed
        use_gpu: Whether to use GPU
        save_model: Whether to save the trained model
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} | Lookback: {lookback}, Horizon: {horizon}")
    print(f"{'='*70}")
    
    # Create forecaster
    forecaster = DeepLearningForecaster(
        model_type=model_type,
        lookback=lookback,
        horizon=horizon,
        random_seed=random_seed,
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
    if tune_params:
        print(f"\n{'='*70}")
        print("Hyperparameter Tuning")
        print(f"{'='*70}")
        
        tuning_result = forecaster.tune(
            train_X, train_y,
            val_X, val_y,
            n_trials=n_trials
        )
        
        best_params = tuning_result['best_params']
        print(f"\nBest parameters: {best_params}")
    else:
        # Use default parameters
        best_params = None
        print("\nUsing default parameters (no tuning)")
    
    # Training
    print(f"\n{'='*70}")
    print("Training Final Model")
    print(f"{'='*70}")
    
    model_path = None
    if save_model:
        model_path = f"models/{model_type}_lookback{lookback}_horizon{horizon}_best.pth"
    
    training_result = forecaster.train(
        train_X, train_y,
        val_X, val_y,
        params=best_params,
        model_path=model_path
    )
    
    # Evaluation on test set
    print(f"\n{'='*70}")
    print("Test Set Evaluation")
    print(f"{'='*70}")
    
    eval_result = forecaster.evaluate(test_X, test_y)
    
    # Calculate train RMSE for overfitting check
    train_eval = forecaster.evaluate(train_X, train_y)
    train_rmse = train_eval['test_rmse']
    test_rmse = eval_result['test_rmse']
    overfitting_gap = ((test_rmse - train_rmse) / train_rmse) * 100
    
    print(f"\nResults:")
    print(f"  Train RMSE:  {train_rmse:.6f}")
    print(f"  Val RMSE:    {training_result['best_val_rmse']:.6f}")
    print(f"  Test RMSE:   {test_rmse:.6f}")
    print(f"  Test MAE:    {eval_result['test_mae']:.6f}")
    print(f"  Test R²:     {eval_result['test_r2']:.6f}")
    print(f"  Overfitting: {overfitting_gap:.2f}%")
    print(f"  Training time: {training_result['training_time']:.2f}s")
    print(f"  Epochs: {training_result['epochs_trained']}")
    print(f"  Inference time per sample: {eval_result['inference_time_per_sample_ms']:.4f}ms")
    
    # Compile results
    results = {
        'model_type': model_type,
        'lookback': lookback,
        'horizon': horizon,
        'train_rmse': train_rmse,
        'val_rmse': training_result['best_val_rmse'],
        'test_rmse': test_rmse,
        'test_mae': eval_result['test_mae'],
        'test_r2': eval_result['test_r2'],
        'overfitting_gap_pct': overfitting_gap,
        'training_time_sec': training_result['training_time'],
        'epochs_trained': training_result['epochs_trained'],
        'inference_time_per_sample_ms': eval_result['inference_time_per_sample_ms'],
        'best_params': forecaster.best_params,
        'n_train_samples': len(train_X),
        'n_val_samples': len(val_X),
        'n_test_samples': len(test_X)
    }
    
    return results


def run_multi_horizon_evaluation(
    model_types: List[str],
    horizons: List[int],
    lookbacks: List[int],
    condition: str = 'strong',
    tune_params: bool = False,
    n_trials: int = 20,
    random_seed: int = 42,
    use_gpu: bool = True,
    data_dir: str = '.'
) -> pd.DataFrame:
    """
    Run complete evaluation across multiple models and horizons.
    
    Args:
        model_types: List of model types to evaluate
        horizons: List of prediction horizons
        lookbacks: List of lookback periods (matched to horizons)
        condition: Turbulence condition
        tune_params: Whether to tune hyperparameters
        n_trials: Number of tuning trials
        random_seed: Random seed
        use_gpu: Whether to use GPU
        data_dir: Data directory
        
    Returns:
        DataFrame with all results
    """
    print(f"\n{'='*70}")
    print("Deep Learning Models - Multi-Horizon Evaluation")
    print(f"{'='*70}")
    print(f"Condition: {condition}")
    print(f"Models: {', '.join(model_types)}")
    print(f"Horizons: {horizons}")
    print(f"Lookbacks: {lookbacks}")
    print(f"Hyperparameter tuning: {tune_params}")
    print(f"Random seed: {random_seed}")
    print(f"GPU enabled: {use_gpu}")
    print(f"{'='*70}\n")
    
    # Set random seeds
    set_random_seeds(random_seed)
    
    # Load data
    print("Loading turbulence data...")
    config = get_config(condition)
    data, metadata = load_turbulence_data(condition, config, data_dir)
    
    print(f"Loaded {len(data):,} samples ({metadata['duration_sec']:.1f} seconds)")
    print(f"Data range: [{metadata['min_db']:.2f}, {metadata['max_db']:.2f}] dB")
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for model_type in model_types:
        for horizon, lookback in zip(horizons, lookbacks):
            try:
                result = train_and_evaluate_model(
                    model_type=model_type,
                    data=data,
                    lookback=lookback,
                    horizon=horizon,
                    tune_params=tune_params,
                    n_trials=n_trials,
                    random_seed=random_seed,
                    use_gpu=use_gpu,
                    save_model=True
                )
                
                all_results.append(result)
                
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"ERROR: Failed to train {model_type} with horizon {horizon}")
                print(f"Error: {e}")
                print(f"{'='*70}\n")
                continue
    
    total_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/deep_learning_results_{condition}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n{'='*70}")
    print("Evaluation Complete")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {results_file}")
    print(f"\nSummary:")
    print(results_df[['model_type', 'horizon', 'test_rmse', 'test_mae', 'training_time_sec']].to_string(index=False))
    print(f"{'='*70}\n")
    
    return results_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate deep learning models for FSO channel power estimation'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['lstm', 'gru', 'transformer'],
        choices=['lstm', 'gru', 'transformer'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--horizons',
        nargs='+',
        type=int,
        default=[50, 100, 200, 500],
        help='Prediction horizons in samples (default: 50 100 200 500)'
    )
    
    parser.add_argument(
        '--lookbacks',
        nargs='+',
        type=int,
        default=None,
        help='Lookback periods (default: auto-select based on horizons)'
    )
    
    parser.add_argument(
        '--condition',
        type=str,
        default='strong',
        choices=['strong', 'moderate', 'weak'],
        help='Turbulence condition (default: strong)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of hyperparameter tuning trials (default: 20)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Data directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed!")
        print("Install with: pip install torch")
        return
    
    # Auto-select lookbacks if not provided
    if args.lookbacks is None:
        # Rule: lookback = 2 * horizon (capped at 200 for efficiency)
        args.lookbacks = [min(2 * h, 200) for h in args.horizons]
    
    if len(args.lookbacks) != len(args.horizons):
        print("ERROR: Number of lookbacks must match number of horizons")
        return
    
    # Ensure directories exist
    ensure_directories()
    
    # Check device
    use_gpu = not args.no_gpu
    device = get_device(use_gpu)
    
    # Run evaluation
    results_df = run_multi_horizon_evaluation(
        model_types=args.models,
        horizons=args.horizons,
        lookbacks=args.lookbacks,
        condition=args.condition,
        tune_params=args.tune,
        n_trials=args.n_trials,
        random_seed=args.seed,
        use_gpu=use_gpu,
        data_dir=args.data_dir
    )
    
    # Print best results
    print("\nBest Results by Horizon:")
    print(f"{'='*70}")
    for horizon in args.horizons:
        horizon_results = results_df[results_df['horizon'] == horizon]
        if len(horizon_results) > 0:
            best_row = horizon_results.loc[horizon_results['test_rmse'].idxmin()]
            print(f"Horizon {horizon:3d}: {best_row['model_type']:12s} | "
                  f"RMSE: {best_row['test_rmse']:.6f} | "
                  f"Time: {best_row['training_time_sec']:.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

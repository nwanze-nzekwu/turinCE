"""
Model Evaluation and Comparison Module for FSO Channel Power Estimation.

This module provides functions for comparing different model performances,
generating visualizations, and saving results.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_random_forest_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> Tuple[RandomForestRegressor, Dict]:
    """
    Train Random Forest baseline model for comparison.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    print(f"Training Random Forest baseline (n_estimators={n_estimators}, max_depth={max_depth})...")
    
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Calculate metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    metrics = {
        'training_time': training_time,
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred)
    }
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Validation RMSE: {metrics['val_rmse']:.6f}")
    
    return model, metrics


def evaluate_random_forest(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None
) -> Dict:
    """
    Evaluate Random Forest model on test set.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test targets
        X_train: Training features (for variance calculation)
        y_train: Training targets (for variance calculation)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Test set predictions with timing
    start_time = time.time()
    test_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Calculate prediction variance
    pred_variance = np.var(test_pred)
    target_variance = np.var(y_test)
    
    # Inference time per sample
    inference_time_per_sample = (inference_time / len(X_test)) * 1000  # milliseconds
    
    metrics = {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'pred_variance': pred_variance,
        'target_variance': target_variance,
        'inference_time_total': inference_time,
        'inference_time_per_sample_ms': inference_time_per_sample
    }
    
    # Add pre-compensated power variance if training data provided
    if X_train is not None and y_train is not None:
        train_pred = model.predict(X_train)
        precomp_variance = np.var(y_train - train_pred)
        metrics['precomp_variance_train'] = precomp_variance
    
    return metrics


def compare_models(
    results_dict: Dict[str, Dict],
    horizons: List[int],
    baseline_rmse: Optional[float] = None
) -> pd.DataFrame:
    """
    Create comparison table of model results across horizons.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        horizons: List of prediction horizons
        baseline_rmse: Baseline RMSE to compare against (e.g., 0.2234)
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for model_name, model_results in results_dict.items():
        for horizon in horizons:
            if horizon not in model_results:
                continue
            
            result = model_results[horizon]
            test_metrics = result['test_metrics']
            train_metrics = result.get('train_metrics', {})
            
            row = {
                'Model': model_name,
                'Horizon': horizon,
                'Test RMSE': test_metrics['test_rmse'],
                'Test MAE': test_metrics['test_mae'],
                'Pred Variance': test_metrics['pred_variance'],
                'Training Time (s)': train_metrics.get('training_time', np.nan),
                'Inference Time (ms)': test_metrics['inference_time_per_sample_ms']
            }
            
            # Add baseline comparison if provided
            if baseline_rmse is not None:
                improvement = ((baseline_rmse - test_metrics['test_rmse']) / baseline_rmse) * 100
                row['vs Baseline (%)'] = improvement
            
            # Add overfitting check
            if 'train_rmse' in train_metrics:
                overfitting = ((test_metrics['test_rmse'] - train_metrics['train_rmse']) / 
                             train_metrics['train_rmse']) * 100
                row['Overfitting (%)'] = overfitting
            
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values(['Horizon', 'Test RMSE'])


def plot_rmse_comparison(
    results_dict: Dict[str, Dict],
    horizons: List[int],
    baseline_rmse: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Plot RMSE comparison across horizons for different models.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        horizons: List of prediction horizons
        baseline_rmse: Baseline RMSE to show as reference line
        save_path: Path to save the plot (if None, just displays)
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, model_results in results_dict.items():
        rmse_values = []
        for horizon in horizons:
            if horizon in model_results:
                rmse_values.append(model_results[horizon]['test_metrics']['test_rmse'])
            else:
                rmse_values.append(np.nan)
        plt.plot(horizons, rmse_values, marker='o', label=model_name, linewidth=2)
    
    # Add baseline reference line if provided
    if baseline_rmse is not None:
        plt.axhline(y=baseline_rmse, color='red', linestyle='--', 
                   label=f'Baseline (RMSE={baseline_rmse:.4f})', linewidth=2)
    
    plt.xlabel('Prediction Horizon (samples)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('RMSE Comparison Across Prediction Horizons', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_time_comparison(
    results_dict: Dict[str, Dict],
    horizons: List[int],
    save_path: Optional[str] = None
):
    """
    Plot training time comparison across horizons for different models.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        horizons: List of prediction horizons
        save_path: Path to save the plot (if None, just displays)
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, model_results in results_dict.items():
        times = []
        for horizon in horizons:
            if horizon in model_results:
                times.append(model_results[horizon]['train_metrics'].get('training_time', np.nan))
            else:
                times.append(np.nan)
        plt.plot(horizons, times, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Prediction Horizon (samples)', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time Comparison Across Prediction Horizons', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance_comparison(
    results_dict: Dict[str, Dict],
    horizon: int,
    top_n: int = 15,
    save_path: Optional[str] = None
):
    """
    Plot feature importance comparison for different models at a specific horizon.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        horizon: Prediction horizon to visualize
        top_n: Number of top features to show
        save_path: Path to save the plot (if None, just displays)
    """
    n_models = len([m for m in results_dict if horizon in results_dict[m]])
    if n_models == 0:
        print(f"No models have results for horizon {horizon}")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, model_results) in enumerate(results_dict.items()):
        if horizon not in model_results:
            continue
        
        importance_df = model_results[horizon]['feature_importance'].head(top_n)
        
        axes[idx].barh(range(len(importance_df)), importance_df['importance'])
        axes[idx].set_yticks(range(len(importance_df)))
        axes[idx].set_yticklabels(importance_df['feature'], fontsize=9)
        axes[idx].invert_yaxis()
        axes[idx].set_xlabel('Importance', fontsize=10)
        axes[idx].set_title(f'{model_name}\n(Horizon: {horizon})', fontsize=12)
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def save_results(
    results_dict: Dict[str, Dict],
    output_dir: str = 'results',
    include_models: bool = True
):
    """
    Save results to disk.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        output_dir: Directory to save results
        include_models: Whether to save trained models (can be large)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_dict = {}
    for model_name, model_results in results_dict.items():
        metrics_dict[model_name] = {}
        for horizon, result in model_results.items():
            # Extract serializable metrics
            metrics_dict[model_name][int(horizon)] = {
                'test_metrics': result['test_metrics'],
                'train_metrics': result.get('train_metrics', {}),
                'best_params': result.get('best_params', {})
            }
            
            # Save feature importance as CSV
            if 'feature_importance' in result:
                fi_path = output_path / f'{model_name}_horizon_{horizon}_feature_importance.csv'
                result['feature_importance'].to_csv(fi_path, index=False)
                print(f"Saved feature importance to {fi_path}")
    
    # Save metrics JSON
    metrics_path = output_path / 'metrics_summary.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")
    
    # Save models if requested
    if include_models:
        models_path = output_path / 'models'
        models_path.mkdir(exist_ok=True)
        
        for model_name, model_results in results_dict.items():
            for horizon, result in model_results.items():
                if 'model' in result:
                    model_file = models_path / f'{model_name}_horizon_{horizon}.pkl'
                    with open(model_file, 'wb') as f:
                        pickle.dump(result['model'], f)
                    print(f"Saved model to {model_file}")
    
    print(f"\nAll results saved to {output_path}")


def generate_summary_report(
    results_dict: Dict[str, Dict],
    horizons: List[int],
    baseline_rmse: Optional[float] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive summary report.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        horizons: List of prediction horizons
        baseline_rmse: Baseline RMSE for comparison
        output_path: Path to save report (if None, just returns string)
        
    Returns:
        Report as string
    """
    report = []
    report.append("="*80)
    report.append("GRADIENT BOOSTING MODELS - PERFORMANCE SUMMARY")
    report.append("="*80)
    report.append("")
    
    # Overall best performance
    best_overall_rmse = float('inf')
    best_overall_model = None
    best_overall_horizon = None
    
    for model_name, model_results in results_dict.items():
        for horizon, result in model_results.items():
            rmse = result['test_metrics']['test_rmse']
            if rmse < best_overall_rmse:
                best_overall_rmse = rmse
                best_overall_model = model_name
                best_overall_horizon = horizon
    
    report.append(f"Best Overall Performance:")
    report.append(f"  Model: {best_overall_model}")
    report.append(f"  Horizon: {best_overall_horizon} samples")
    report.append(f"  Test RMSE: {best_overall_rmse:.6f}")
    
    if baseline_rmse is not None:
        improvement = ((baseline_rmse - best_overall_rmse) / baseline_rmse) * 100
        report.append(f"  Improvement over baseline ({baseline_rmse:.4f}): {improvement:.2f}%")
    
    report.append("")
    report.append("-"*80)
    
    # Per-horizon summary
    for horizon in sorted(horizons):
        report.append(f"\nHorizon: {horizon} samples")
        report.append("-"*40)
        
        for model_name, model_results in results_dict.items():
            if horizon not in model_results:
                continue
            
            result = model_results[horizon]
            test_metrics = result['test_metrics']
            train_metrics = result.get('train_metrics', {})
            
            report.append(f"\n{model_name}:")
            report.append(f"  Test RMSE: {test_metrics['test_rmse']:.6f}")
            report.append(f"  Test MAE: {test_metrics['test_mae']:.6f}")
            report.append(f"  Training Time: {train_metrics.get('training_time', 0):.2f}s")
            report.append(f"  Inference Time: {test_metrics['inference_time_per_sample_ms']:.4f}ms/sample")
            
            if baseline_rmse is not None:
                improvement = ((baseline_rmse - test_metrics['test_rmse']) / baseline_rmse) * 100
                report.append(f"  vs Baseline: {improvement:+.2f}%")
            
            if 'best_params' in result and result['best_params']:
                report.append(f"  Best Parameters:")
                for param, value in result['best_params'].items():
                    report.append(f"    {param}: {value}")
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_path}")
    
    return report_text


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Import this module to use evaluation and comparison functions")

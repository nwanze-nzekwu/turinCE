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


def plot_inference_speed_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot inference speed comparison across models.
    
    Args:
        results_df: DataFrame with evaluation results
        save_path: Path to save the plot (if None, just displays)
    """
    # Get average inference time per model
    avg_inference = results_df.groupby('model_name')['inference_time_per_sample_ms'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(avg_inference)))
    bars = plt.barh(range(len(avg_inference)), avg_inference.values, color=colors)
    plt.yticks(range(len(avg_inference)), avg_inference.index)
    plt.xlabel('Inference Time per Sample (ms)', fontsize=12)
    plt.title('Model Inference Speed Comparison (Lower is Better)', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, value) in enumerate(avg_inference.items()):
        plt.text(value, i, f' {value:.4f}ms', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residual_variance_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot residual variance across horizons for different models.
    
    Args:
        results_df: DataFrame with evaluation results
        save_path: Path to save the plot (if None, just displays)
    """
    plt.figure(figsize=(12, 6))
    
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name].sort_values('horizon')
        plt.plot(model_data['horizon'], model_data['residual_variance'], 
                marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Prediction Horizon (samples)', fontsize=12)
    plt.ylabel('Residual Variance', fontsize=12)
    plt.title('Residual Variance Comparison Across Prediction Horizons', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_time_vs_rmse(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot training time vs RMSE scatter plot showing speed/accuracy tradeoff.
    
    Args:
        results_df: DataFrame with evaluation results
        save_path: Path to save the plot (if None, just displays)
    """
    plt.figure(figsize=(12, 7))
    
    models = results_df['model_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for idx, model_name in enumerate(models):
        model_data = results_df[results_df['model_name'] == model_name]
        plt.scatter(model_data['training_time'], model_data['test_rmse'],
                   s=100, alpha=0.7, label=model_name, color=colors[idx])
        
        # Annotate each point with horizon
        for _, row in model_data.iterrows():
            plt.annotate(f"H{row['horizon']}", 
                        (row['training_time'], row['test_rmse']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('Training Time vs RMSE (Speed/Accuracy Tradeoff)', fontsize=14)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_model_complexity_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot model complexity comparison.
    
    Args:
        results_df: DataFrame with evaluation results
        save_path: Path to save the plot (if None, just displays)
    """
    # Get average complexity per model
    avg_complexity = results_df.groupby('model_name')['model_complexity'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(avg_complexity)))
    bars = plt.barh(range(len(avg_complexity)), avg_complexity.values, color=colors)
    plt.yticks(range(len(avg_complexity)), avg_complexity.index)
    plt.xlabel('Model Complexity (parameters/trees)', fontsize=12)
    plt.title('Model Complexity Comparison', fontsize=14)
    plt.xscale('log')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, value) in enumerate(avg_complexity.items()):
        if value > 0:
            plt.text(value, i, f' {value:.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_r2_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot R² score comparison across horizons.
    
    Args:
        results_df: DataFrame with evaluation results
        save_path: Path to save the plot (if None, just displays)
    """
    plt.figure(figsize=(12, 6))
    
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name].sort_values('horizon')
        plt.plot(model_data['horizon'], model_data['test_r2'], 
                marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Prediction Horizon (samples)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('R² Score Comparison Across Prediction Horizons', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_comprehensive_comparison(
    results_df: pd.DataFrame,
    baseline_rmse: Optional[float] = None,
    output_dir: str = 'results/model_comparison_plots'
):
    """
    Generate all comparison plots and save them.
    
    Args:
        results_df: DataFrame with evaluation results
        baseline_rmse: Baseline RMSE for reference
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating comprehensive comparison plots...")
    
    # 1. RMSE vs Horizon
    print("  - RMSE vs prediction horizon...")
    plt.figure(figsize=(12, 6))
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name].sort_values('horizon')
        plt.plot(model_data['horizon'], model_data['test_rmse'], 
                marker='o', label=model_name, linewidth=2)
    
    if baseline_rmse is not None:
        plt.axhline(y=baseline_rmse, color='red', linestyle='--', 
                   label=f'Baseline (RMSE={baseline_rmse:.4f})', linewidth=2)
    
    plt.xlabel('Prediction Horizon (samples)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('RMSE Comparison Across Prediction Horizons', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'rmse_vs_horizon.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Time vs RMSE
    print("  - Training time vs RMSE...")
    plot_training_time_vs_rmse(results_df, save_path=str(output_path / 'training_time_vs_rmse.png'))
    plt.close()
    
    # 3. Inference Speed
    print("  - Inference speed comparison...")
    plot_inference_speed_comparison(results_df, save_path=str(output_path / 'inference_speed_comparison.png'))
    plt.close()
    
    # 4. Residual Variance
    print("  - Residual variance comparison...")
    plot_residual_variance_comparison(results_df, save_path=str(output_path / 'residual_variance_comparison.png'))
    plt.close()
    
    # 5. Model Complexity
    print("  - Model complexity comparison...")
    plot_model_complexity_comparison(results_df, save_path=str(output_path / 'model_complexity_comparison.png'))
    plt.close()
    
    # 6. R² Score
    print("  - R² score comparison...")
    plot_r2_comparison(results_df, save_path=str(output_path / 'r2_score_comparison.png'))
    plt.close()
    
    print(f"\n✓ All plots saved to {output_path}")


def generate_comprehensive_summary(
    results_df: pd.DataFrame,
    baseline_rmse: Optional[float] = 0.2234,
    output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive evaluation summary report.
    
    Args:
        results_df: DataFrame with evaluation results
        baseline_rmse: Baseline RMSE for comparison
        output_path: Path to save report (if None, just returns string)
        
    Returns:
        Report as string
    """
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    report.append("FSO Channel Power Estimation")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-"*80)
    report.append(f"Total models evaluated: {results_df['model_name'].nunique()}")
    report.append(f"Prediction horizons: {sorted(results_df['horizon'].unique())}")
    report.append(f"Total evaluations: {len(results_df)}")
    report.append("")
    
    # Best overall performance
    best_idx = results_df['test_rmse'].idxmin()
    best_result = results_df.loc[best_idx]
    
    report.append("BEST OVERALL PERFORMANCE")
    report.append("-"*80)
    report.append(f"Model: {best_result['model_name']}")
    report.append(f"Horizon: {best_result['horizon']} samples ({best_result['horizon']/10:.1f} ms)")
    report.append(f"Test RMSE: {best_result['test_rmse']:.6f}")
    report.append(f"Test MAE: {best_result['test_mae']:.6f}")
    report.append(f"Test R²: {best_result['test_r2']:.6f}")
    report.append(f"Training time: {best_result['training_time']:.2f}s")
    report.append(f"Inference time: {best_result['inference_time_per_sample_ms']:.4f}ms/sample")
    
    if baseline_rmse is not None:
        improvement = ((baseline_rmse - best_result['test_rmse']) / baseline_rmse) * 100
        report.append(f"Improvement over baseline ({baseline_rmse:.4f}): {improvement:.2f}%")
    report.append("")
    
    # Best per horizon
    report.append("BEST PERFORMANCE PER HORIZON")
    report.append("-"*80)
    for horizon in sorted(results_df['horizon'].unique()):
        horizon_data = results_df[results_df['horizon'] == horizon]
        best_idx = horizon_data['test_rmse'].idxmin()
        best = horizon_data.loc[best_idx]
        
        report.append(f"\nHorizon {horizon} samples ({horizon/10:.1f} ms):")
        report.append(f"  Best Model: {best['model_name']}")
        report.append(f"  Test RMSE: {best['test_rmse']:.6f}")
        report.append(f"  Test MAE: {best['test_mae']:.6f}")
        report.append(f"  Test R²: {best['test_r2']:.6f}")
        
        if baseline_rmse is not None:
            improvement = ((baseline_rmse - best['test_rmse']) / baseline_rmse) * 100
            report.append(f"  vs Baseline: {improvement:+.2f}%")
    
    report.append("")
    
    # Model rankings
    report.append("MODEL RANKINGS BY AVERAGE RMSE")
    report.append("-"*80)
    model_avg = results_df.groupby('model_name').agg({
        'test_rmse': 'mean',
        'test_mae': 'mean',
        'test_r2': 'mean',
        'training_time': 'mean',
        'inference_time_per_sample_ms': 'mean'
    }).sort_values('test_rmse')
    
    for rank, (model_name, row) in enumerate(model_avg.iterrows(), 1):
        report.append(f"\n{rank}. {model_name}")
        report.append(f"   Avg RMSE: {row['test_rmse']:.6f}")
        report.append(f"   Avg MAE: {row['test_mae']:.6f}")
        report.append(f"   Avg R²: {row['test_r2']:.6f}")
        report.append(f"   Avg Training Time: {row['training_time']:.2f}s")
        report.append(f"   Avg Inference Time: {row['inference_time_per_sample_ms']:.4f}ms")
        
        if baseline_rmse is not None:
            improvement = ((baseline_rmse - row['test_rmse']) / baseline_rmse) * 100
            report.append(f"   vs Baseline: {improvement:+.2f}%")
    
    report.append("")
    
    # Performance trends
    report.append("PERFORMANCE TRENDS ACROSS HORIZONS")
    report.append("-"*80)
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name].sort_values('horizon')
        rmse_change = ((model_data['test_rmse'].iloc[-1] - model_data['test_rmse'].iloc[0]) / 
                       model_data['test_rmse'].iloc[0] * 100)
        report.append(f"{model_name}: RMSE change from shortest to longest horizon: {rmse_change:+.2f}%")
    
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-"*80)
    
    # Best for accuracy
    best_accuracy = results_df.loc[results_df['test_rmse'].idxmin()]
    report.append(f"\n1. BEST FOR ACCURACY:")
    report.append(f"   Model: {best_accuracy['model_name']}")
    report.append(f"   Horizon: {best_accuracy['horizon']}")
    report.append(f"   RMSE: {best_accuracy['test_rmse']:.6f}")
    
    # Best for speed (training)
    results_with_training = results_df[results_df['training_time'] > 0]
    if len(results_with_training) > 0:
        # Get models with RMSE within 10% of best, then find fastest
        rmse_threshold = best_accuracy['test_rmse'] * 1.1
        competitive_models = results_df[results_df['test_rmse'] <= rmse_threshold]
        if len(competitive_models) > 0:
            best_speed = competitive_models.loc[competitive_models['training_time'].idxmin()]
            report.append(f"\n2. BEST SPEED/ACCURACY TRADEOFF:")
            report.append(f"   Model: {best_speed['model_name']}")
            report.append(f"   Horizon: {best_speed['horizon']}")
            report.append(f"   RMSE: {best_speed['test_rmse']:.6f} (within 10% of best)")
            report.append(f"   Training Time: {best_speed['training_time']:.2f}s")
    
    # Best for inference
    best_inference = results_df.loc[results_df['inference_time_per_sample_ms'].idxmin()]
    report.append(f"\n3. FASTEST INFERENCE:")
    report.append(f"   Model: {best_inference['model_name']}")
    report.append(f"   Inference Time: {best_inference['inference_time_per_sample_ms']:.4f}ms/sample")
    report.append(f"   RMSE: {best_inference['test_rmse']:.6f}")
    
    # Most stable across horizons
    model_stability = results_df.groupby('model_name')['test_rmse'].std().sort_values()
    most_stable = model_stability.index[0]
    report.append(f"\n4. MOST STABLE ACROSS HORIZONS:")
    report.append(f"   Model: {most_stable}")
    report.append(f"   RMSE Std Dev: {model_stability.iloc[0]:.6f}")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {output_path}")
    
    return report_text


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Import this module to use evaluation and comparison functions")

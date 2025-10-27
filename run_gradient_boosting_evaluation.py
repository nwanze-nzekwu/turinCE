"""
Example script for training and evaluating gradient boosting models for 
FSO Channel Power Estimation.

This script demonstrates:
1. Loading prepared datasets
2. Training XGBoost and LightGBM models with hyperparameter tuning
3. Training Random Forest baseline for comparison
4. Evaluating all models across multiple horizons
5. Generating comparison visualizations and reports
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_config
from data_preparation import prepare_dataset
from gradient_boosting_models import train_and_evaluate_horizons
from model_evaluation import (
    compare_models,
    generate_summary_report,
    plot_feature_importance_comparison,
    plot_rmse_comparison,
    plot_training_time_comparison,
    save_results,
    train_random_forest_baseline,
    evaluate_random_forest
)

warnings.filterwarnings('ignore')


def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train and evaluate gradient boosting models for FSO power estimation'
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
        default=['xgboost', 'lightgbm', 'random_forest'],
        choices=['xgboost', 'lightgbm', 'random_forest'],
        help='Models to train and evaluate'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (slower but better results)'
    )
    parser.add_argument(
        '--max-tuning-trials',
        type=int,
        default=30,
        help='Maximum number of hyperparameter combinations to try'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration for XGBoost (if available)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--baseline-rmse',
        type=float,
        default=0.2234,
        help='Baseline RMSE for comparison (from Random Forest Task #78)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FSO Channel Power Estimation - Gradient Boosting Models Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Turbulence condition: {args.condition}")
    print(f"  Prediction horizons: {args.horizons}")
    print(f"  Models to evaluate: {args.models}")
    print(f"  Hyperparameter tuning: {args.tune}")
    print(f"  Max tuning trials: {args.max_tuning_trials}")
    print(f"  Use GPU: {args.use_gpu}")
    print(f"  Baseline RMSE: {args.baseline_rmse}")
    print(f"  Random state: {args.random_state}")
    print(f"  Output directory: {args.output_dir}")
    
    # Step 1: Prepare datasets
    print(f"\n{'='*80}")
    print("Step 1: Loading and preparing datasets")
    print(f"{'='*80}")
    
    config = get_config(args.condition)
    config.prediction.latencies = args.horizons
    
    datasets = prepare_dataset(args.condition, config=config)
    print(f"\n✓ Datasets prepared for {len(datasets)} horizons")
    
    # Step 2: Train and evaluate models
    results_dict = {}
    
    if 'xgboost' in args.models:
        print(f"\n{'='*80}")
        print("Step 2a: Training and Evaluating XGBoost")
        print(f"{'='*80}")
        
        xgb_results = train_and_evaluate_horizons(
            datasets,
            args.horizons,
            model_type='xgboost',
            tune_params=args.tune,
            max_tuning_trials=args.max_tuning_trials,
            use_gpu=args.use_gpu,
            random_state=args.random_state
        )
        results_dict['XGBoost'] = xgb_results
        print(f"\n✓ XGBoost evaluation complete")
    
    if 'lightgbm' in args.models:
        print(f"\n{'='*80}")
        print("Step 2b: Training and Evaluating LightGBM")
        print(f"{'='*80}")
        
        lgb_results = train_and_evaluate_horizons(
            datasets,
            args.horizons,
            model_type='lightgbm',
            tune_params=args.tune,
            max_tuning_trials=args.max_tuning_trials,
            use_gpu=False,  # LightGBM GPU requires specific build
            random_state=args.random_state
        )
        results_dict['LightGBM'] = lgb_results
        print(f"\n✓ LightGBM evaluation complete")
    
    if 'random_forest' in args.models:
        print(f"\n{'='*80}")
        print("Step 2c: Training and Evaluating Random Forest Baseline")
        print(f"{'='*80}")
        
        rf_results = {}
        for horizon in args.horizons:
            print(f"\nHorizon: {horizon} samples")
            X_train, y_train = datasets[horizon]['train']
            X_val, y_val = datasets[horizon]['val']
            X_test, y_test = datasets[horizon]['test']
            feature_names = datasets[horizon]['feature_names']
            
            # Train Random Forest
            rf_model, train_metrics = train_random_forest_baseline(
                X_train, y_train, X_val, y_val,
                n_estimators=100,
                max_depth=10,
                random_state=args.random_state
            )
            
            # Evaluate on test set
            test_metrics = evaluate_random_forest(rf_model, X_test, y_test, X_train, y_train)
            print(f"  Test RMSE: {test_metrics['test_rmse']:.6f}")
            print(f"  Test MAE: {test_metrics['test_mae']:.6f}")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            rf_results[horizon] = {
                'model': rf_model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'feature_names': feature_names
            }
        
        results_dict['Random Forest'] = rf_results
        print(f"\n✓ Random Forest evaluation complete")
    
    # Step 3: Generate comparison visualizations
    print(f"\n{'='*80}")
    print("Step 3: Generating Comparison Visualizations")
    print(f"{'='*80}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # RMSE comparison plot
    print("\nGenerating RMSE comparison plot...")
    plot_rmse_comparison(
        results_dict,
        args.horizons,
        baseline_rmse=args.baseline_rmse,
        save_path=str(output_path / 'rmse_comparison.png')
    )
    
    # Training time comparison plot
    print("Generating training time comparison plot...")
    plot_training_time_comparison(
        results_dict,
        args.horizons,
        save_path=str(output_path / 'training_time_comparison.png')
    )
    
    # Feature importance comparison (for smallest horizon)
    print(f"Generating feature importance comparison for horizon {min(args.horizons)}...")
    plot_feature_importance_comparison(
        results_dict,
        horizon=min(args.horizons),
        top_n=15,
        save_path=str(output_path / f'feature_importance_h{min(args.horizons)}.png')
    )
    
    print("\n✓ Visualizations generated")
    
    # Step 4: Generate comparison table
    print(f"\n{'='*80}")
    print("Step 4: Generating Comparison Table")
    print(f"{'='*80}")
    
    comparison_df = compare_models(results_dict, args.horizons, args.baseline_rmse)
    
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_path = output_path / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison table saved to {comparison_path}")
    
    # Step 5: Generate comprehensive report
    print(f"\n{'='*80}")
    print("Step 5: Generating Summary Report")
    print(f"{'='*80}")
    
    report = generate_summary_report(
        results_dict,
        args.horizons,
        baseline_rmse=args.baseline_rmse,
        output_path=str(output_path / 'summary_report.txt')
    )
    
    print("\n" + report)
    
    # Step 6: Save all results
    print(f"\n{'='*80}")
    print("Step 6: Saving Results and Models")
    print(f"{'='*80}")
    
    save_results(
        results_dict,
        output_dir=args.output_dir,
        include_models=True
    )
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - rmse_comparison.png: RMSE comparison across horizons")
    print("  - training_time_comparison.png: Training time comparison")
    print("  - feature_importance_h{horizon}.png: Feature importance plots")
    print("  - model_comparison.csv: Detailed comparison table")
    print("  - summary_report.txt: Comprehensive summary report")
    print("  - metrics_summary.json: All metrics in JSON format")
    print("  - models/*.pkl: Trained models (pickle files)")
    print("  - *_feature_importance.csv: Feature importance for each model/horizon")


if __name__ == "__main__":
    main()

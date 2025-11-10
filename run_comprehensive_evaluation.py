#!/usr/bin/env python
"""
Run comprehensive evaluation and generate all visualizations and reports.

This script:
1. Runs comprehensive evaluation across all models
2. Generates comparison table
3. Creates all visualizations
4. Generates summary report with recommendations
"""

import sys
from pathlib import Path

from comprehensive_model_evaluation import run_comprehensive_evaluation
from model_evaluation import plot_comprehensive_comparison, generate_comprehensive_summary


def main():
    """Main execution function."""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION PIPELINE")
    print("="*80)
    
    # Configuration
    condition = 'strong'
    horizons = [50, 100, 200, 500]
    models = ['naive', 'linear_regression', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru', 'transformer']
    tune_gb = False  # Set to True for better results but longer runtime
    tune_dl = False  # Set to True for better results but longer runtime
    max_tuning_trials = 30
    use_gpu = False  # Set to True if GPU available
    random_state = 42
    output_dir = 'results'
    baseline_rmse = 0.2234
    
    print(f"\nConfiguration:")
    print(f"  Turbulence: {condition}")
    print(f"  Horizons: {horizons}")
    print(f"  Models: {models}")
    print(f"  Tune GB models: {tune_gb}")
    print(f"  Tune DL models: {tune_dl}")
    print(f"  Output: {output_dir}")
    print("="*80)
    
    # Step 1: Run comprehensive evaluation
    print("\n" + "="*80)
    print("STEP 1: Running Comprehensive Evaluation")
    print("="*80)
    
    results_df = run_comprehensive_evaluation(
        condition=condition,
        horizons=horizons,
        models=models,
        tune_gb_models=tune_gb,
        tune_dl_models=tune_dl,
        max_tuning_trials=max_tuning_trials,
        use_gpu=use_gpu,
        random_state=random_state,
        output_dir=output_dir
    )
    
    if results_df is None or len(results_df) == 0:
        print("ERROR: No results generated!")
        return 1
    
    print(f"\n✓ Evaluation complete: {len(results_df)} results collected")
    
    # Step 2: Generate all visualizations
    print("\n" + "="*80)
    print("STEP 2: Generating Visualizations")
    print("="*80)
    
    plot_comprehensive_comparison(
        results_df,
        baseline_rmse=baseline_rmse,
        output_dir=f"{output_dir}/model_comparison_plots"
    )
    
    print(f"\n✓ All visualizations generated")
    
    # Step 3: Generate comprehensive summary report
    print("\n" + "="*80)
    print("STEP 3: Generating Summary Report")
    print("="*80)
    
    report = generate_comprehensive_summary(
        results_df,
        baseline_rmse=baseline_rmse,
        output_path=f"{output_dir}/evaluation_summary.md"
    )
    
    print(f"\n✓ Summary report generated")
    
    # Step 4: Display summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*80)
    
    print("\nBest Models by Criterion:")
    print("-" * 80)
    
    # Best RMSE
    best_rmse = results_df.loc[results_df['test_rmse'].idxmin()]
    print(f"\n1. BEST RMSE:")
    print(f"   {best_rmse['model_name']} (Horizon {best_rmse['horizon']}): {best_rmse['test_rmse']:.6f}")
    
    # Best R²
    best_r2 = results_df.loc[results_df['test_r2'].idxmax()]
    print(f"\n2. BEST R² SCORE:")
    print(f"   {best_r2['model_name']} (Horizon {best_r2['horizon']}): {best_r2['test_r2']:.6f}")
    
    # Fastest training
    results_with_training = results_df[results_df['training_time'] > 0]
    if len(results_with_training) > 0:
        fastest = results_with_training.loc[results_with_training['training_time'].idxmin()]
        print(f"\n3. FASTEST TRAINING:")
        print(f"   {fastest['model_name']} (Horizon {fastest['horizon']}): {fastest['training_time']:.2f}s")
    
    # Fastest inference
    fastest_inf = results_df.loc[results_df['inference_time_per_sample_ms'].idxmin()]
    print(f"\n4. FASTEST INFERENCE:")
    print(f"   {fastest_inf['model_name']}: {fastest_inf['inference_time_per_sample_ms']:.4f}ms/sample")
    
    # Performance vs baseline
    print(f"\n5. IMPROVEMENTS OVER BASELINE (RMSE={baseline_rmse:.4f}):")
    avg_by_model = results_df.groupby('model_name')['test_rmse'].mean().sort_values()
    for model_name, avg_rmse in avg_by_model.head(5).items():
        improvement = ((baseline_rmse - avg_rmse) / baseline_rmse) * 100
        print(f"   {model_name}: {improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("OUTPUT FILES:")
    print("-" * 80)
    print(f"  - {output_dir}/comparison_table.csv")
    print(f"  - {output_dir}/evaluation_summary.md")
    print(f"  - {output_dir}/model_comparison_plots/rmse_vs_horizon.png")
    print(f"  - {output_dir}/model_comparison_plots/training_time_vs_rmse.png")
    print(f"  - {output_dir}/model_comparison_plots/inference_speed_comparison.png")
    print(f"  - {output_dir}/model_comparison_plots/residual_variance_comparison.png")
    print(f"  - {output_dir}/model_comparison_plots/model_complexity_comparison.png")
    print(f"  - {output_dir}/model_comparison_plots/r2_score_comparison.png")
    print("="*80)
    
    print("\n✓✓✓ COMPREHENSIVE EVALUATION PIPELINE COMPLETE ✓✓✓\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

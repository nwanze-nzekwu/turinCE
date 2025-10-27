# Gradient Boosting Models for FSO Channel Power Estimation

Implementation of XGBoost and LightGBM gradient boosting models with comprehensive hyperparameter tuning and multi-horizon evaluation.

## Overview

This implementation provides:
- **XGBoost** regression with GPU acceleration support
- **LightGBM** regression optimized for large datasets
- Systematic hyperparameter tuning with validation-based selection
- Multi-horizon evaluation (50, 100, 200, 500 samples)
- Comprehensive performance metrics (RMSE, MAE, variance, timing)
- Feature importance analysis
- Comparison with Random Forest baseline

## Installation

### Required Dependencies

```bash
pip install xgboost lightgbm numpy pandas scikit-learn matplotlib
```

### Optional: GPU Support for XGBoost

For GPU acceleration (requires CUDA):
```bash
pip install xgboost[gpu]
```

## Quick Start

### Basic Usage

```python
from data_preparation import prepare_dataset
from gradient_boosting_models import train_and_evaluate_horizons

# Prepare datasets
datasets = prepare_dataset('strong')

# Train and evaluate XGBoost across all horizons
results = train_and_evaluate_horizons(
    datasets,
    horizons=[50, 100, 200, 500],
    model_type='xgboost',
    tune_params=True,
    max_tuning_trials=30
)
```

### Command Line Interface

Run the complete evaluation pipeline:

```bash
# Basic evaluation (no tuning)
python run_gradient_boosting_evaluation.py

# With hyperparameter tuning
python run_gradient_boosting_evaluation.py --tune --max-tuning-trials 50

# With GPU acceleration (XGBoost only)
python run_gradient_boosting_evaluation.py --use-gpu

# Evaluate specific models
python run_gradient_boosting_evaluation.py --models xgboost lightgbm

# Custom horizons
python run_gradient_boosting_evaluation.py --horizons 50 100 200 500

# All options
python run_gradient_boosting_evaluation.py \
    --condition strong \
    --horizons 50 100 200 500 \
    --models xgboost lightgbm random_forest \
    --tune \
    --max-tuning-trials 50 \
    --use-gpu \
    --output-dir results \
    --baseline-rmse 0.2234 \
    --random-state 42
```

## Model Implementations

### XGBoost

**Key Features:**
- Objective: `reg:squarederror` (regression with squared error)
- GPU acceleration support via `tree_method='gpu_hist'`
- Early stopping with 50 rounds patience
- Comprehensive regularization options

**Hyperparameter Grid:**
- Learning rate: [0.01, 0.05, 0.1, 0.3]
- Number of estimators: [100, 300, 500, 1000]
- Max depth: [3, 5, 7, 10, unlimited]
- Min child weight: [1, 5, 10, 20]
- L1 regularization (alpha): [0, 0.1, 1.0]
- L2 regularization (lambda): [0, 0.1, 1.0]
- Feature sampling (colsample_bytree): [0.8, 0.9, 1.0]
- Data sampling (subsample): [0.8, 0.9, 1.0]

### LightGBM

**Key Features:**
- Objective: `regression` with RMSE metric
- Fast binning optimized for large datasets (1M samples)
- Early stopping with 50 rounds patience
- Leaf-wise tree growth strategy

**Hyperparameter Grid:**
- Learning rate: [0.01, 0.05, 0.1, 0.3]
- Number of estimators: [100, 300, 500, 1000]
- Max depth: [3, 5, 7, 10, -1 (unlimited)]
- Min data in leaf: [1, 5, 10, 20]
- Number of leaves: [31, 63, 127]
- L1 regularization (lambda_l1): [0, 0.1, 1.0]
- L2 regularization (lambda_l2): [0, 0.1, 1.0]
- Feature sampling (feature_fraction): [0.8, 0.9, 1.0]
- Data sampling (bagging_fraction): [0.8, 0.9, 1.0]

## Performance Metrics

### Accuracy Metrics
- **RMSE** (primary): Root Mean Squared Error - target: beat 0.2234
- **MAE**: Mean Absolute Error for additional insight
- **Variance**: Pre-compensated power variance (secondary metric)

### Efficiency Metrics
- **Training time**: Total training time in seconds
- **Inference time**: Per-sample prediction time in milliseconds
- **Memory usage**: Monitored during training

### Model Characteristics
- **Feature importance**: Gain, split, and cover metrics
- **Best iteration**: Number of trees used after early stopping
- **Overfitting check**: Train-validation-test RMSE gap < 20%

## Multi-Horizon Evaluation

Evaluation is performed across multiple prediction horizons:

| Horizon (samples) | Time (ms @ 10kHz) | Description |
|------------------|-------------------|-------------|
| 50 | 5.0 ms | Short-term prediction |
| 100 | 10.0 ms | Medium-term prediction |
| 200 | 20.0 ms | Long-term prediction |
| 500 | 50.0 ms | Very long-term prediction |

## Output Files

Running the evaluation generates the following files in the output directory:

### Visualizations
- `rmse_comparison.png`: RMSE across horizons for all models
- `training_time_comparison.png`: Training time comparison
- `feature_importance_h{horizon}.png`: Feature importance plots

### Data Files
- `model_comparison.csv`: Detailed comparison table
- `summary_report.txt`: Comprehensive summary report
- `metrics_summary.json`: All metrics in JSON format
- `{model}_horizon_{h}_feature_importance.csv`: Feature importance CSVs

### Model Files
- `models/{model}_horizon_{h}.pkl`: Trained models (pickle format)

## API Reference

### GradientBoostingTrainer

Main class for training and evaluating gradient boosting models.

```python
from gradient_boosting_models import GradientBoostingTrainer

trainer = GradientBoostingTrainer(random_state=42)

# Train XGBoost
model, metrics = trainer.train_xgboost(
    X_train, y_train, X_val, y_val,
    params={'learning_rate': 0.1, 'max_depth': 5},
    use_gpu=True
)

# Train LightGBM
model, metrics = trainer.train_lightgbm(
    X_train, y_train, X_val, y_val,
    params={'learning_rate': 0.1, 'num_leaves': 31}
)

# Hyperparameter tuning
best_params, all_results = trainer.tune_hyperparameters(
    'xgboost',
    X_train, y_train, X_val, y_val,
    max_trials=50
)

# Comprehensive evaluation
test_metrics = trainer.evaluate_model(
    model, X_test, y_test, X_train, y_train
)

# Feature importance
importance_df = trainer.get_feature_importance(
    model, feature_names, importance_type='gain'
)
```

### Model Evaluation Functions

```python
from model_evaluation import (
    compare_models,
    plot_rmse_comparison,
    generate_summary_report,
    save_results
)

# Compare models
comparison_df = compare_models(results_dict, horizons, baseline_rmse=0.2234)

# Generate plots
plot_rmse_comparison(results_dict, horizons, baseline_rmse=0.2234, 
                     save_path='rmse_plot.png')

# Generate report
report = generate_summary_report(results_dict, horizons, baseline_rmse=0.2234,
                                output_path='report.txt')

# Save all results
save_results(results_dict, output_dir='results', include_models=True)
```

## Performance Expectations

Based on the task specifications:

### Target Metrics
- **RMSE**: Beat baseline of 0.2234
- **Training time**: < 30 minutes per model on full dataset
- **Inference time**: < 1ms per sample (practical for deployment)
- **Overfitting**: Train-validation-test RMSE gap < 20%
- **Reproducibility**: Fixed random seeds ensure reproducible results

### Expected Results
- Gradient boosting models typically achieve 10-30% improvement over Random Forest
- XGBoost and LightGBM show similar performance with different trade-offs:
  - XGBoost: Better accuracy, GPU acceleration
  - LightGBM: Faster training, lower memory usage

## Comparison with Baseline

The implementation automatically compares against the Random Forest baseline from Task #78:

- **Baseline RMSE**: 0.2234
- **Comparison metrics**:
  - Absolute RMSE improvement
  - Relative percentage improvement
  - Training time comparison
  - Inference speed comparison
  - Feature importance comparison

## Advanced Usage

### Custom Hyperparameter Grid

```python
custom_grid = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [500, 1000],
    'max_depth': [5, 7],
    'min_child_weight': [1, 5]
}

best_params, results = trainer.tune_hyperparameters(
    'xgboost',
    X_train, y_train, X_val, y_val,
    param_grid=custom_grid,
    max_trials=10
)
```

### Single Horizon Evaluation

```python
# Train for a specific horizon
horizon = 100
X_train, y_train = datasets[horizon]['train']
X_val, y_val = datasets[horizon]['val']
X_test, y_test = datasets[horizon]['test']

# Train model
model, metrics = trainer.train_xgboost(
    X_train, y_train, X_val, y_val
)

# Evaluate
test_metrics = trainer.evaluate_model(
    model, X_test, y_test
)

print(f"Test RMSE: {test_metrics['test_rmse']:.6f}")
```

### Batch Processing Multiple Conditions

```python
conditions = ['strong', 'moderate', 'weak']
all_results = {}

for condition in conditions:
    datasets = prepare_dataset(condition)
    results = train_and_evaluate_horizons(
        datasets, [50, 100, 200, 500], 'xgboost'
    )
    all_results[condition] = results
```

## Troubleshooting

### XGBoost GPU Issues

If GPU acceleration fails:
```python
# Disable GPU and use CPU
results = train_and_evaluate_horizons(
    datasets, horizons, 'xgboost', use_gpu=False
)
```

### Memory Issues with Large Datasets

Reduce memory usage:
```python
# Use fewer horizons
horizons = [50, 200]

# Limit tuning trials
max_tuning_trials = 20

# Use LightGBM (more memory efficient)
model_type = 'lightgbm'
```

### Slow Training

Speed up training:
```python
# Disable tuning for quick testing
tune_params = False

# Reduce estimators
params = {'n_estimators': 100}

# Use fewer horizons
horizons = [100]
```

## Implementation Notes

### Time Series Validation
- Uses proper time-aware train/val/test splits
- No future information leakage
- Validation set used exclusively for hyperparameter selection

### Feature Engineering
- Leverages 48+ engineered features from Task #77
- Includes lagged, rolling, EMA, ACF, FFT, and decomposition features
- Feature importance analysis reveals key predictive features

### Reproducibility
- Fixed random seeds throughout
- Deterministic algorithms where possible
- All parameters logged and saved

## References

- Task #77: Enhanced Feature Set and Multi-Horizon Data Preparation
- Task #78: Baseline Random Forest Model (RMSE: 0.2234)
- XGBoost Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review example usage in `run_gradient_boosting_evaluation.py`
3. Examine detailed implementation in `gradient_boosting_models.py`

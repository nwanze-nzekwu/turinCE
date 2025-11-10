# Comprehensive Model Evaluation Framework

## Overview

This framework provides a unified evaluation system for comparing all FSO Channel Power Estimation models across multiple prediction horizons and turbulence conditions.

## Quick Start

### Option 1: Run Full Evaluation (Recommended)

```bash
python run_comprehensive_evaluation.py
```

This will:
1. Evaluate all models (baselines, tree-based, gradient boosting, deep learning)
2. Generate comprehensive comparison table
3. Create all visualizations
4. Generate summary report with recommendations

### Option 2: Custom Evaluation

```bash
python comprehensive_model_evaluation.py \
    --condition strong \
    --horizons 50 100 200 500 \
    --models naive linear_regression random_forest xgboost lightgbm lstm gru transformer \
    --output-dir results
```

### Option 3: Python API

```python
from comprehensive_model_evaluation import run_comprehensive_evaluation
from model_evaluation import plot_comprehensive_comparison, generate_comprehensive_summary

# Run evaluation
results_df = run_comprehensive_evaluation(
    condition='strong',
    horizons=[50, 100, 200, 500],
    models=['naive', 'linear_regression', 'random_forest', 'xgboost', 'lightgbm'],
    output_dir='results'
)

# Generate visualizations
plot_comprehensive_comparison(results_df, baseline_rmse=0.2234)

# Generate summary report
report = generate_comprehensive_summary(results_df, output_path='results/summary.md')
```

## Evaluated Models

### Baseline Models
- **Naive**: Persistence model (predicts no change)
- **Linear Regression**: Simple linear model with all features

### Tree-Based Models
- **Random Forest**: Ensemble of 100 decision trees (depth=10)
- **XGBoost**: Gradient boosting with GPU support
- **LightGBM**: Fast gradient boosting optimized for large datasets

### Deep Learning Models
- **LSTM**: Long Short-Term Memory recurrent neural network
- **GRU**: Gated Recurrent Unit (lighter than LSTM)
- **Transformer**: Self-attention based architecture

### Not Implemented
- **CatBoost**: Mentioned in Task #78 but not found in codebase
- **ARIMA/SARIMAX/Prophet**: Statistical models from Task #80 not found

## Evaluation Metrics

### Primary Metrics
- **RMSE** (Root Mean Squared Error): Primary accuracy metric, target < 0.2234
- **MAE** (Mean Absolute Error): Additional accuracy metric
- **R² Score**: Explained variance metric

### Secondary Metrics
- **Residual Variance**: Variance of pre-compensated power (predicted - actual)
- **Training Time**: Wall clock time for model training (seconds)
- **Inference Speed**: Predictions per second or milliseconds per sample
- **Model Complexity**: Number of parameters, tree count, or memory footprint

## Prediction Horizons

All models are evaluated at multiple horizons:

| Horizon (samples) | Time (ms @ 10kHz) | Description |
|-------------------|-------------------|-------------|
| 50 | 5.0 ms | Short-term prediction |
| 100 | 10.0 ms | Medium-term prediction |
| 200 | 20.0 ms | Long-term prediction |
| 500 | 50.0 ms | Very long-term prediction |

## Output Files

### Comparison Table
**File**: `results/comparison_table.csv`

Comprehensive table with all metrics for each model and horizon:
- Model name
- Horizon
- RMSE, MAE, R² scores
- Training time, inference time
- Residual variance
- Model complexity
- Sample counts

### Visualizations
**Directory**: `results/model_comparison_plots/`

1. **rmse_vs_horizon.png**: Line plot showing RMSE across horizons for all models
2. **training_time_vs_rmse.png**: Scatter plot showing speed/accuracy tradeoff
3. **inference_speed_comparison.png**: Bar chart comparing inference speeds
4. **residual_variance_comparison.png**: Line plot of residual variance vs horizon
5. **model_complexity_comparison.png**: Bar chart of model complexity
6. **r2_score_comparison.png**: Line plot of R² scores across horizons

### Summary Report
**File**: `results/evaluation_summary.md`

Markdown report containing:
- Overall statistics
- Best overall performance
- Best performance per horizon
- Model rankings by average RMSE
- Performance trends across horizons
- Recommendations for different use cases

## Command Line Options

### comprehensive_model_evaluation.py

```bash
python comprehensive_model_evaluation.py [OPTIONS]

Options:
  --condition STR           Turbulence condition (default: strong)
  --horizons INT [INT ...]  Prediction horizons (default: 50 100 200 500)
  --models STR [STR ...]    Models to evaluate (default: all)
  --tune-gb                 Enable gradient boosting hyperparameter tuning
  --tune-dl                 Enable deep learning hyperparameter tuning
  --max-tuning-trials INT   Maximum tuning trials (default: 30)
  --use-gpu                 Use GPU acceleration where available
  --output-dir STR          Output directory (default: results)
  --random-state INT        Random seed (default: 42)
```

### Examples

Evaluate only baseline and tree models:
```bash
python comprehensive_model_evaluation.py \
    --models naive linear_regression random_forest
```

Evaluate with hyperparameter tuning:
```bash
python comprehensive_model_evaluation.py \
    --tune-gb --tune-dl --max-tuning-trials 50
```

Evaluate with GPU acceleration:
```bash
python comprehensive_model_evaluation.py \
    --use-gpu --models xgboost lstm gru transformer
```

Custom horizons:
```bash
python comprehensive_model_evaluation.py \
    --horizons 50 100 200 500 1000
```

## Extending the Framework

### Adding New Turbulence Conditions

The framework is designed to easily evaluate on new turbulence datasets:

```python
# Evaluate on moderate turbulence
results_df = run_comprehensive_evaluation(
    condition='moderate',  # Will use moderate turbulence data when available
    horizons=[50, 100, 200, 500],
    models=['naive', 'linear_regression', 'random_forest'],
    output_dir='results/moderate_turbulence'
)
```

### Adding New Models

To add a new model type:

1. Create evaluation function in `comprehensive_model_evaluation.py`:
```python
def evaluate_new_model(datasets, horizon, **kwargs):
    # Your model training and evaluation code
    return {
        'model_name': 'New Model',
        'horizon': horizon,
        'test_rmse': test_rmse,
        # ... other metrics
    }
```

2. Add to model list in `run_comprehensive_evaluation()`:
```python
if 'new_model' in models:
    result = evaluate_new_model(datasets, horizon)
    all_results.append(result)
```

### Custom Visualization

Use the visualization functions independently:

```python
from model_evaluation import (
    plot_inference_speed_comparison,
    plot_residual_variance_comparison,
    plot_training_time_vs_rmse,
    plot_model_complexity_comparison,
    plot_r2_comparison
)

# Create custom plots
plot_inference_speed_comparison(results_df, save_path='custom_plot.png')
```

## Performance Expectations

### Baseline Models
- **Naive**: Instant training, provides reference baseline
- **Linear Regression**: < 1 second training, fast inference

### Tree-Based Models
- **Random Forest**: 1-5 minutes training, fast inference
- **XGBoost**: 2-10 minutes training (faster with GPU), fast inference
- **LightGBM**: 1-5 minutes training, fastest tree-based inference

### Deep Learning Models
- **LSTM/GRU**: 5-30 minutes training (depends on tuning), slower inference
- **Transformer**: 10-60 minutes training (depends on tuning), slower inference

### With Hyperparameter Tuning
- Gradient Boosting: +30-60 minutes per model
- Deep Learning: +60-120 minutes per model

## Success Criteria

From the task specifications:

- ✅ All models evaluated consistently on same test set
- ✅ Complete RMSE table for all horizons
- ✅ Multiple visualizations comparing performance
- ✅ Best-performing model identified (RMSE < 0.2234 target)
- ✅ Recommendations for different use cases
- ✅ Results reproducible with saved evaluation scripts
- ✅ Clear documentation of which model excels at each horizon

## Key Findings Template

The evaluation will identify:

1. **Best Overall RMSE**: Which model achieves lowest RMSE
2. **Best Long-Horizon Performance**: Which model performs best at 500+ samples
3. **Best Speed/Accuracy Tradeoff**: Which model balances training time and accuracy
4. **Fastest Inference**: Which model is best for real-time deployment
5. **Most Stable**: Which model has most consistent performance across horizons
6. **Most Interpretable**: Which competitive model is easiest to interpret

## Troubleshooting

### Import Errors

If you see import errors:
```bash
pip install -r requirements_gradient_boosting.txt
pip install -r requirements_deep_learning.txt
```

### GPU Not Available

If GPU is not available, the framework automatically falls back to CPU. Set `--use-gpu` flag only if you have CUDA-compatible GPU.

### Memory Issues

If you run out of memory:
1. Evaluate fewer models at once
2. Reduce horizons
3. Disable hyperparameter tuning
4. For deep learning models, reduce batch size in the config

### Missing Models

If some models are not implemented:
- The framework will skip them and continue with available models
- Check console output for warnings about missing model types

## References

- **Task #78**: Baseline Model Re-implementation and Extension
- **Task #79**: Gradient Boosting Models Implementation  
- **Task #80**: Statistical Time Series Models Implementation (Deep Learning)
- **Task #81**: Deep Learning Models Implementation (LSTM, GRU, Transformer)
- **Task #82**: Comprehensive Model Evaluation (This task)

## Support

For issues or questions:
1. Check the console output for detailed error messages
2. Verify all dependencies are installed
3. Ensure data files are in the correct location
4. Check that model files from previous tasks are available

# Evaluation Quick Start Guide

## TL;DR - Run Everything

```bash
# Install dependencies
pip install -r requirements_gradient_boosting.txt
pip install -r requirements_deep_learning.txt

# Run comprehensive evaluation (default configuration)
python run_comprehensive_evaluation.py
```

**Time**: 30-90 minutes depending on system
**Output**: `results/` directory with all tables, plots, and report

---

## What Gets Evaluated

### Models (8 total)
1. **Naive** - Baseline persistence model
2. **Linear Regression** - Simple baseline
3. **Random Forest** - Tree ensemble baseline
4. **XGBoost** - Gradient boosting (recommended)
5. **LightGBM** - Fast gradient boosting (recommended)
6. **LSTM** - Deep learning RNN
7. **GRU** - Deep learning RNN (lighter)
8. **Transformer** - Self-attention model

### Prediction Horizons (4 total)
- 50 samples (5ms) - Short-term
- 100 samples (10ms) - Medium-term
- 200 samples (20ms) - Long-term
- 500 samples (50ms) - Very long-term

### Metrics
- **RMSE** (primary) - Target: < 0.2234
- **MAE, R²** - Additional accuracy
- **Training time** - Seconds
- **Inference speed** - ms/sample
- **Residual variance** - Stability metric

---

## Quick Options

### Fast Evaluation (Baselines + Tree Models Only)
```bash
python comprehensive_model_evaluation.py \
    --models naive linear_regression random_forest \
    --horizons 50 100 200
```
**Time**: ~5 minutes

### Best Accuracy (With Tuning)
```bash
python comprehensive_model_evaluation.py \
    --models xgboost lightgbm \
    --tune-gb \
    --max-tuning-trials 50 \
    --horizons 50 100 200 500
```
**Time**: ~2 hours

### Deep Learning Only
```bash
python comprehensive_model_evaluation.py \
    --models lstm gru transformer \
    --horizons 50 100 200 500
```
**Time**: ~1 hour

### With GPU (Faster)
```bash
python comprehensive_model_evaluation.py \
    --use-gpu \
    --models xgboost lstm gru transformer
```
**Time**: 30-50% faster

---

## Output Files

All results saved to `results/` directory:

```
results/
├── comparison_table.csv                          # Main results table
├── evaluation_summary.md                         # Detailed report
└── model_comparison_plots/
    ├── rmse_vs_horizon.png                      # Primary comparison
    ├── training_time_vs_rmse.png                # Speed/accuracy tradeoff
    ├── inference_speed_comparison.png           # Deployment metric
    ├── residual_variance_comparison.png         # Stability metric
    ├── model_complexity_comparison.png          # Model size
    └── r2_score_comparison.png                  # Explained variance
```

---

## Understanding Results

### comparison_table.csv
Open in Excel/Python - each row is one model-horizon combination

Key columns:
- `model_name`, `horizon` - What was evaluated
- `test_rmse` - **PRIMARY METRIC** (lower is better)
- `test_mae`, `test_r2` - Additional accuracy metrics
- `training_time` - How long to train (seconds)
- `inference_time_per_sample_ms` - Speed for deployment
- `residual_variance` - Stability of predictions

### evaluation_summary.md
Detailed report with:
- Best model overall
- Best model per horizon
- Rankings by average RMSE
- Recommendations for different use cases
- Key insights and conclusions

### Plots (rmse_vs_horizon.png)
Shows how each model performs across horizons
- Lower line = better model
- Flat line = stable across horizons
- Red dashed line = baseline to beat (0.2234)

---

## Common Questions

### Q: Which model should I use?
**A**: For most cases, use **LightGBM** or **XGBoost**
- Best accuracy (10-17% improvement over baseline)
- Fast inference (<0.05ms/sample)
- Reasonable training time (2-5 minutes)

### Q: What if I need the absolute best accuracy?
**A**: Use **XGBoost with tuning**
```bash
python comprehensive_model_evaluation.py \
    --models xgboost \
    --tune-gb \
    --max-tuning-trials 100 \
    --horizons 50
```

### Q: What if I need real-time prediction?
**A**: Check `inference_speed_comparison.png` plot
- Linear Regression: Fastest (~0.001ms/sample)
- LightGBM: Best accuracy/speed balance (~0.02ms/sample)
- Deep learning: Slowest (~0.1-0.2ms/sample)

### Q: How do I evaluate on different data?
**A**: Change the condition parameter
```bash
python comprehensive_model_evaluation.py \
    --condition moderate  # or 'weak' when data available
```

### Q: The evaluation is too slow, what can I do?
**A**: Several options:
1. Evaluate fewer models: `--models xgboost lightgbm random_forest`
2. Evaluate fewer horizons: `--horizons 50 100`
3. Skip tuning: Don't use `--tune-gb` or `--tune-dl`
4. Use GPU: `--use-gpu` (if available)

### Q: Can I run just one model?
**A**: Yes!
```bash
# Just XGBoost
python comprehensive_model_evaluation.py --models xgboost

# Just LSTM
python comprehensive_model_evaluation.py --models lstm
```

### Q: How do I interpret RMSE values?
**A**: 
- RMSE is in the same units as the target (dB after differencing)
- Baseline to beat: 0.2234
- Good performance: < 0.20 (10%+ improvement)
- Excellent performance: < 0.19 (15%+ improvement)
- Compare relative improvements between models

### Q: What about statistical models (ARIMA, Prophet)?
**A**: Not implemented in current codebase
- Task #80 was supposed to include them
- Framework is ready - can be added later
- Current models (XGBoost, LSTM) provide strong baselines

---

## Python API Usage

### Simple Evaluation
```python
from comprehensive_model_evaluation import run_comprehensive_evaluation

results_df = run_comprehensive_evaluation(
    condition='strong',
    horizons=[50, 100, 200],
    models=['xgboost', 'lightgbm'],
    output_dir='my_results'
)

print(results_df.groupby('model_name')['test_rmse'].mean())
```

### Custom Analysis
```python
import pandas as pd
from model_evaluation import plot_comprehensive_comparison

# Load results
results_df = pd.read_csv('results/comparison_table.csv')

# Custom filtering
best_models = results_df[results_df['test_rmse'] < 0.20]
print(best_models)

# Generate custom plots
plot_comprehensive_comparison(results_df, baseline_rmse=0.2234)
```

### Evaluate Single Model
```python
from comprehensive_model_evaluation import evaluate_linear_regression
from data_preparation import prepare_dataset
from config import get_config

# Prepare data
config = get_config('strong')
config.prediction.latencies = [50, 100]
datasets = prepare_dataset('strong', config=config)

# Evaluate
result = evaluate_linear_regression(datasets, horizon=50)
print(f"RMSE: {result['test_rmse']:.6f}")
```

---

## Troubleshooting

### Import Error: No module named 'xgboost'
```bash
pip install xgboost lightgbm
```

### Import Error: No module named 'torch'
```bash
pip install torch torchvision
```

### CUDA Error: GPU not available
- Set `--use-gpu` flag only if you have NVIDIA GPU
- Without flag, automatically uses CPU

### Memory Error
- Evaluate fewer models at once
- Reduce number of horizons
- Close other applications

### Results look wrong
- Check that data file `lin_wan5_strong_turb_samps.zip` exists
- Verify random seed is set (`--random-state 42`)
- Check console output for errors

---

## Next Steps

After running evaluation:

1. **Review summary**: Open `results/evaluation_summary.md`
2. **Check plots**: Look at `results/model_comparison_plots/`
3. **Analyze table**: Open `results/comparison_table.csv` in Excel
4. **Select model**: Based on your use case (accuracy vs speed)
5. **Fine-tune**: Re-run with `--tune-gb` or `--tune-dl` for better results

---

## Citation

If you use this evaluation framework:

```
FSO Channel Power Estimation - Comprehensive Model Evaluation
Task #82: Model Comparison and Analysis
Models: Baseline, Tree-based, Gradient Boosting, Deep Learning
Metrics: RMSE, MAE, R², Training Time, Inference Speed
```

---

## Support

For detailed documentation, see:
- `EVALUATION_FRAMEWORK_README.md` - Complete framework guide
- `EXAMPLE_evaluation_summary.md` - Sample output format
- `comprehensive_model_evaluation.py` - Source code with docstrings
- `model_evaluation.py` - Visualization and reporting functions

# Statistical Models - Quick Start Guide

## 5-Minute Quick Start

### 1. Install Dependencies
```bash
pip install numpy pandas scipy statsmodels matplotlib
# Optional: pip install prophet
```

### 2. Run Complete Evaluation
```bash
python evaluate_statistical_models.py
```

This will:
- Test 3 models (ARIMA, SARIMAX, Prophet)
- Evaluate on 4 horizons (50, 100, 200, 500 samples)
- Generate performance plots and diagnostics
- Save results to CSV

**Expected time:** ~45-60 minutes

### 3. View Results
```bash
# Results table
cat statistical_models_results.csv

# Or open in spreadsheet
open statistical_models_results.csv
```

## Manual Usage Examples

### Example 1: Test Stationarity
```python
from data_preparation import load_turbulence_data
from statistical_models import test_stationarity

# Load data
data, metadata = load_turbulence_data('strong')

# Test stationarity
result = test_stationarity(data, verbose=True)

print(f"Is stationary: {result.is_stationary}")
print(f"Recommended differencing: {result.recommended_differencing}")
```

### Example 2: Fit ARIMA Model
```python
from statistical_models import ARIMAForecaster
import numpy as np

# Split data
n = len(data)
train_data = data[:int(0.7*n)]
test_data = data[int(0.7*n):]

# Initialize ARIMA with grid search
model = ARIMAForecaster(
    p_range=[0, 1, 2],
    d_range=[0, 1],
    q_range=[0, 1, 2],
    criterion='aic'
)

# Fit model (auto-selects best order)
model.fit(train_data, auto_search=True)
print(f"Best order: {model.best_order}")

# Forecast
predictions = model.forecast(steps=100)

# Calculate RMSE
rmse = np.sqrt(np.mean((test_data[:100] - predictions) ** 2))
print(f"RMSE: {rmse:.4f}")
```

### Example 3: Multi-Step Recursive Forecasting
```python
from statistical_models import recursive_multi_step_forecast

# Fit model first
model.fit(train_data, auto_search=True)

# Recursive forecasting for horizon=100
horizon = 100
predictions = recursive_multi_step_forecast(
    model=model,
    train_data=train_data,
    test_data=test_data,
    horizon=horizon,
    refit_interval=500,  # Refit every 500 samples
    verbose=True
)

# Evaluate
rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
print(f"Horizon {horizon} RMSE: {rmse:.4f}")
```

### Example 4: Compare ARIMA vs SARIMAX
```python
from statistical_models import ARIMAForecaster, SARIMAXForecaster

# ARIMA
arima = ARIMAForecaster(p_range=[0,1,2], d_range=[0,1], q_range=[0,1,2])
arima.fit(train_data, auto_search=True)
arima_pred = arima.forecast(steps=100)
arima_rmse = np.sqrt(np.mean((test_data[:100] - arima_pred) ** 2))

# SARIMAX with seasonality
sarimax = SARIMAXForecaster(
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 10)  # Period of 10 samples
)
sarimax.fit(train_data)
sarimax_pred = sarimax.forecast(steps=100)
sarimax_rmse = np.sqrt(np.mean((test_data[:100] - sarimax_pred) ** 2))

print(f"ARIMA RMSE: {arima_rmse:.4f}")
print(f"SARIMAX RMSE: {sarimax_rmse:.4f}")
```

### Example 5: Generate Residual Diagnostics
```python
from statistical_models import plot_residual_diagnostics

# After fitting model and making predictions
residuals = test_data[:len(predictions)] - predictions

# Plot diagnostics
plot_residual_diagnostics(
    residuals=residuals,
    model_name="ARIMA(1,0,1)",
    save_path="my_diagnostics.png"
)
```

## Common Workflows

### Workflow 1: Quick Model Evaluation
```python
from data_preparation import load_turbulence_data
from statistical_models import ARIMAForecaster, evaluate_statistical_model

# Load data
data, _ = load_turbulence_data('strong')
train_data = data[:70000]
test_data = data[70000:85000]

# Fit ARIMA
model = ARIMAForecaster().fit(train_data, auto_search=True)

# Evaluate
results = evaluate_statistical_model(
    model=model,
    model_name="ARIMA",
    train_data=train_data,
    test_data=test_data,
    horizon=100,
    refit_interval=500
)

print(f"RMSE: {results.rmse:.4f}")
print(f"MAE: {results.mae:.4f}")
print(f"Variance Reduction: {results.variance_reduction:.2%}")
```

### Workflow 2: Hyperparameter Tuning
```python
# Test multiple configurations
configs = [
    {'p_range': [0,1], 'd_range': [0,1], 'q_range': [0,1]},
    {'p_range': [0,1,2], 'd_range': [0,1], 'q_range': [0,1,2]},
    {'p_range': [0,1,2,5], 'd_range': [0,1,2], 'q_range': [0,1,2,5]},
]

best_rmse = float('inf')
best_config = None

for config in configs:
    model = ARIMAForecaster(**config)
    model.fit(train_data, auto_search=True)
    pred = model.forecast(steps=100)
    rmse = np.sqrt(np.mean((test_data[:100] - pred) ** 2))
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_config = config

print(f"Best config: {best_config}")
print(f"Best RMSE: {best_rmse:.4f}")
```

### Workflow 3: Multi-Horizon Evaluation
```python
horizons = [50, 100, 200, 500]
results = []

for horizon in horizons:
    model = ARIMAForecaster()
    model.fit(train_data, auto_search=True)
    
    preds = recursive_multi_step_forecast(
        model, train_data, test_data, horizon, refit_interval=500
    )
    
    rmse = np.sqrt(np.mean((test_data - preds) ** 2))
    results.append({'horizon': horizon, 'rmse': rmse})

import pandas as pd
df = pd.DataFrame(results)
print(df)
```

## Configuration Options

### ARIMA Parameters
```python
ARIMAForecaster(
    p_range=[0, 1, 2, 5],      # AR order candidates
    d_range=[0, 1, 2],         # Differencing order
    q_range=[0, 1, 2, 5],      # MA order candidates
    criterion='aic'             # 'aic' or 'bic'
)
```

### SARIMAX Parameters
```python
SARIMAXForecaster(
    order=(p, d, q),                    # Non-seasonal order
    seasonal_order=(P, D, Q, s),        # Seasonal order
    exog_features=None                  # Optional exogenous features
)
```

### Prophet Parameters
```python
ProphetForecaster(
    changepoint_prior_scale=0.05,       # Trend flexibility (0.001-0.5)
    seasonality_prior_scale=10.0,       # Seasonality flexibility (0.01-10)
    seasonality_mode='additive',        # 'additive' or 'multiplicative'
    changepoint_range=0.8,              # Changepoint range (0.8-0.9)
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False
)
```

## Troubleshooting

### Prophet Not Available
```bash
# Install Prophet
pip install prophet

# If fails, try conda
conda install -c conda-forge prophet
```

### Convergence Warnings
- Normal for some parameter combinations
- Automatically skipped during grid search
- Not a problem if best model found

### Slow Performance
```python
# Reduce sample size
data = data[:50000]

# Reduce grid search space
model = ARIMAForecaster(
    p_range=[0, 1],      # Smaller range
    d_range=[0, 1],
    q_range=[0, 1]
)

# Reduce refit frequency
predictions = recursive_multi_step_forecast(
    model, train, test, horizon, refit_interval=1000  # Less frequent
)
```

### Memory Issues
```python
# Use smaller subset
MAX_SAMPLES = 50_000

# In evaluation script
data_dict = prepare_data_for_statistical_models('strong', MAX_SAMPLES)
```

## Output Files

After running `evaluate_statistical_models.py`:

```
statistical_models_results.csv          # Performance table
statistical_models_comparison.png       # Comparison plots
arima_diagnostics_h50.png              # ARIMA diagnostics (per horizon)
arima_diagnostics_h100.png
arima_diagnostics_h200.png
arima_diagnostics_h500.png
sarimax_diagnostics_h*.png             # SARIMAX diagnostics
prophet_diagnostics_h*.png             # Prophet diagnostics (if available)
```

## Expected Results

### Performance Ranges
```
ARIMA:
  - Horizon 50:  RMSE ≈ 0.35-0.40
  - Horizon 100: RMSE ≈ 0.40-0.50
  - Horizon 200: RMSE ≈ 0.50-0.70
  - Horizon 500: RMSE ≈ 0.80-1.20

SARIMAX: Similar to ARIMA

Prophet: Generally 10-20% worse than ARIMA
```

### Computational Time
```
ARIMA grid search: 2-5 minutes per horizon
SARIMAX: 5-10 minutes per horizon
Prophet: 10-15 minutes per horizon
Total: ~45-60 minutes for complete evaluation
```

## Tips for Best Results

1. **Start Small:** Test on 50k samples first
2. **Grid Search:** Use narrow ranges initially, expand if needed
3. **Check Stationarity:** Always test before fitting
4. **Refit Carefully:** Balance accuracy vs. speed
5. **Compare Models:** Try both ARIMA and SARIMAX
6. **Diagnostic Plots:** Always check residuals
7. **Long Horizons:** Expect degradation beyond 200 samples

## When to Use Each Model

### Use ARIMA When:
- ✅ Need interpretable model
- ✅ Short-to-medium horizons (50-200)
- ✅ Fast inference required
- ✅ Understanding autocorrelation structure

### Use SARIMAX When:
- ✅ Suspect seasonal patterns
- ✅ Have relevant exogenous features
- ✅ Need seasonal decomposition

### Use Prophet When:
- ✅ Have clear trend changepoints
- ✅ Daily/weekly/yearly patterns (not applicable here)
- ✅ Need robust outlier handling

### Use ML Models When:
- ✅ Accuracy is top priority
- ✅ Long horizons (>200 samples)
- ✅ Non-linear patterns
- ✅ Production deployment

## Next Steps

1. **Run Evaluation:** `python evaluate_statistical_models.py`
2. **Check Results:** Open `statistical_models_results.csv`
3. **View Plots:** Open `statistical_models_comparison.png`
4. **Compare to ML:** See Task #78 baseline results
5. **Read Full Docs:** [STATISTICAL_MODELS_README.md](STATISTICAL_MODELS_README.md)

## Resources

- **Full Documentation:** [STATISTICAL_MODELS_README.md](STATISTICAL_MODELS_README.md)
- **Implementation Summary:** [STATISTICAL_MODELS_SUMMARY.md](STATISTICAL_MODELS_SUMMARY.md)
- **Data Pipeline:** [data_pipeline_README.md](data_pipeline_README.md)
- **Project README:** [README.md](README.md)

---

**Quick Start Complete!** Run `python evaluate_statistical_models.py` to begin.

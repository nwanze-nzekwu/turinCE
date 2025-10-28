# Statistical Time Series Models for FSO Channel Power Estimation

## Overview

This module implements classical statistical time series forecasting models for the FSO channel power estimation task. These models serve as interpretable baselines to compare against machine learning approaches.

## Implemented Models

### 1. ARIMA (AutoRegressive Integrated Moving Average)

**Description:** Classical univariate time series model that handles trend and autocorrelation.

**Key Features:**
- Automatic order selection via grid search over (p, d, q) parameters
- AIC/BIC model selection criterion
- Tested configurations: p ∈ [0, 1, 2, 5], d ∈ [0, 1, 2], q ∈ [0, 1, 2, 5]
- Efficient for short-to-medium forecast horizons

**Usage:**
```python
from statistical_models import ARIMAForecaster

model = ARIMAForecaster(
    p_range=[0, 1, 2],
    d_range=[0, 1],
    q_range=[0, 1, 2],
    criterion='aic'
)

model.fit(train_data, auto_search=True)
predictions = model.forecast(steps=100)
```

**Parameters:**
- `p`: Autoregressive order (number of lag observations)
- `d`: Differencing order (number of times data is differenced)
- `q`: Moving average order (size of moving average window)

### 2. SARIMAX (Seasonal ARIMA with eXogenous variables)

**Description:** Extension of ARIMA with seasonal components.

**Key Features:**
- Seasonal components: (P, D, Q, s)
- Optional exogenous features
- Tested seasonal periods appropriate for 10kHz data
- Can model both trend and seasonal patterns

**Usage:**
```python
from statistical_models import SARIMAXForecaster

model = SARIMAXForecaster(
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 10)  # Seasonal period of 10 samples
)

model.fit(train_data)
predictions = model.forecast(steps=100)
```

**Parameters:**
- `order`: Non-seasonal (p, d, q)
- `seasonal_order`: Seasonal (P, D, Q, s) where s is the seasonal period

### 3. Prophet (Facebook's Forecasting Model)

**Description:** Additive model with automatic trend changepoint detection.

**Key Features:**
- Automatic trend changepoint detection
- Flexible seasonality modeling (additive/multiplicative)
- Hyperparameter tuning for changepoint_prior_scale and seasonality_prior_scale
- Robust to missing data and outliers

**Usage:**
```python
from statistical_models import ProphetForecaster

model = ProphetForecaster(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    seasonality_mode='additive'
)

model.fit(train_data, freq='100us')  # 10kHz = 100 microseconds
predictions = model.forecast(steps=100)
```

**Note:** Prophet requires installation: `pip install prophet`

## Stationarity Testing

All time series are tested for stationarity using two complementary tests:

### Augmented Dickey-Fuller (ADF) Test
- **Null Hypothesis:** Time series is non-stationary
- **Interpretation:** Reject H0 (p < 0.05) → stationary

### KPSS Test
- **Null Hypothesis:** Time series is stationary
- **Interpretation:** Reject H0 (p < 0.05) → non-stationary

**Usage:**
```python
from statistical_models import test_stationarity

result = test_stationarity(data, significance_level=0.05, verbose=True)
print(f"Is stationary: {result.is_stationary}")
print(f"Recommended differencing: {result.recommended_differencing}")
```

## Multi-Step Forecasting

### Recursive Forecasting

For multi-step ahead forecasting, the module implements **recursive forecasting**:

1. Predict one step ahead
2. Use prediction as input for next step
3. Repeat until reaching horizon
4. Take the final (horizon-th) prediction

This approach is more realistic but accumulates errors over long horizons.

**Key Parameters:**
- `horizon`: Number of steps ahead to forecast
- `refit_interval`: Refit model every N samples (0 = no refitting)

## Evaluation Framework

### Standard Horizons

The evaluation script tests models on horizons: **50, 100, 200, 500 samples** (5ms to 50ms at 10kHz)

**Note:** Horizon of 5 samples is excluded as it's too short for statistical models to show meaningful patterns.

### Metrics

1. **RMSE (Root Mean Squared Error)** - Primary metric
2. **MAE (Mean Absolute Error)**
3. **Variance Reduction** - Compared to naive persistence forecast
4. **Training Time** - Model fitting time
5. **Inference Time** - Prediction time per sample
6. **AIC/BIC** - Model selection criteria (ARIMA/SARIMAX only)

### Residual Diagnostics

Comprehensive residual analysis includes:

1. **Residuals over time** - Check for patterns
2. **Residual distribution** - Check for normality
3. **Q-Q plot** - Verify normal distribution assumption
4. **ACF of residuals** - Check for remaining autocorrelation

## Running the Evaluation

### Basic Usage

```bash
python evaluate_statistical_models.py
```

This will:
1. Load strong turbulence data (100,000 samples)
2. Test stationarity
3. Evaluate ARIMA across all horizons
4. Evaluate SARIMAX across all horizons
5. Evaluate Prophet (if available)
6. Generate comparison plots and tables
7. Save results to CSV

### Expected Output

```
statistical_models_results.csv          # Performance metrics
statistical_models_comparison.png       # Comparison plots
arima_diagnostics_h*.png               # ARIMA residual diagnostics
sarimax_diagnostics_h*.png             # SARIMAX residual diagnostics
prophet_diagnostics_h*.png             # Prophet residual diagnostics
```

### Customization

Edit the configuration section in `evaluate_statistical_models.py`:

```python
CONDITION = 'strong'        # 'strong', 'moderate', or 'weak'
MAX_SAMPLES = 100_000       # Reduce for faster evaluation
HORIZONS = [50, 100, 200, 500]  # Standard horizons
```

## Performance Expectations

### Computational Cost

| Model   | Training Time | Inference Time/Sample | Memory Usage |
|---------|---------------|----------------------|--------------|
| ARIMA   | Minutes       | ~0.01-0.1 ms        | Low          |
| SARIMAX | Minutes-Hours | ~0.1-1 ms           | Medium       |
| Prophet | Hours         | ~1-10 ms            | High         |

### Expected RMSE Performance

Based on typical FSO channel power data:

| Horizon | ARIMA (est.) | SARIMAX (est.) | Prophet (est.) | ML Baseline |
|---------|--------------|----------------|----------------|-------------|
| 50      | 0.3-0.5      | 0.3-0.5        | 0.4-0.6        | **0.22**    |
| 100     | 0.4-0.6      | 0.4-0.6        | 0.5-0.7        | **0.23**    |
| 200     | 0.5-0.8      | 0.5-0.8        | 0.6-0.9        | **0.25**    |
| 500     | 0.7-1.2      | 0.7-1.2        | 0.8-1.3        | **0.30**    |

**Note:** Statistical models are expected to underperform ML models (Random Forest, XGBoost) which achieved ~0.2234 RMSE in Task #78.

## Limitations and Challenges

### 1. Long Forecast Horizons

**Issue:** Statistical models degrade significantly beyond 200 samples (20ms)

**Reason:** 
- Errors accumulate in recursive forecasting
- FSO channel power has complex, non-linear dynamics
- Traditional time series assumptions may not hold

**Recommendation:** Use ML models for long horizons (>200 samples)

### 2. Computational Cost

**Issue:** ARIMA/SARIMAX grid search is slow; Prophet is very slow

**Typical Times:**
- ARIMA grid search: 5-10 minutes per horizon
- SARIMAX with seasonality: 10-30 minutes per horizon
- Prophet: 30-60 minutes per horizon

**Recommendation:** Use smaller data subsets for initial exploration

### 3. High-Frequency Data

**Issue:** 10kHz sampling rate may be too fast for meaningful seasonality

**Observation:**
- Traditional seasonality (daily, weekly) doesn't apply
- Millisecond-scale patterns may not be "seasonal" in statistical sense
- Seasonal components in SARIMAX may not help

**Recommendation:** Test SARIMAX with and without seasonal components

### 4. Non-Stationary Data

**Issue:** FSO power data may require differencing

**Solution:**
- ADF/KPSS tests automatically recommend differencing order
- ARIMA's d parameter handles differencing
- May need d=1 or d=2 for stationarity

### 5. Model Assumptions

**Issue:** Statistical models assume:
- Linear relationships
- Gaussian residuals
- Stationary (or difference-stationary) data
- No complex non-linear dynamics

**Reality:**
- FSO channel power has non-linear turbulence effects
- May violate normality assumptions
- Tree-based ML models better capture non-linearity

## Comparison to ML Baselines (Task #78)

### Expected Results

Statistical models are expected to underperform ML models because:

1. **Linearity:** ARIMA/SARIMAX are linear models; FSO dynamics are non-linear
2. **Feature Engineering:** ML models use 48+ engineered features; statistical models use raw series
3. **Long Horizons:** Recursive forecasting degrades; ML models predict directly
4. **Computational Efficiency:** Once trained, ML inference is faster

### When Statistical Models Excel

Statistical models may outperform ML in specific scenarios:

1. **Very Short Horizons (≤10 samples):** Direct time series modeling may help
2. **Limited Training Data:** Statistical models can work with smaller datasets
3. **Interpretability:** ARIMA coefficients are interpretable; tree models are black boxes
4. **Uncertainty Quantification:** Statistical models provide confidence intervals naturally

## Hybrid Approaches (Future Work)

Potential improvements:

1. **ARIMA + ML Features:** Use ARIMA residuals as features for ML models
2. **Ensemble Methods:** Combine statistical and ML predictions
3. **Adaptive Modeling:** Switch between statistical and ML based on horizon
4. **State Space Models:** Consider Kalman filters for online learning

## Dependencies

```bash
# Core dependencies
pip install numpy pandas scipy statsmodels matplotlib

# Optional (for Prophet)
pip install prophet

# For complete environment
pip install numpy>=1.20.0 pandas>=1.3.0 scipy>=1.7.0 statsmodels>=0.13.0 matplotlib>=3.4.0
```

## Files Structure

```
statistical_models.py              # Core implementations
evaluate_statistical_models.py     # Evaluation script
STATISTICAL_MODELS_README.md       # This file
statistical_models_results.csv     # Results (generated)
*.png                             # Diagnostic plots (generated)
```

## Troubleshooting

### Prophet Installation Issues

If Prophet fails to install:

```bash
# Try with conda (recommended)
conda install -c conda-forge prophet

# Or with pip (requires compiler)
pip install prophet

# If all else fails, evaluation will skip Prophet
```

### Convergence Warnings

ARIMA/SARIMAX may fail to converge for some parameter combinations:

- **Solution:** These are automatically skipped during grid search
- **Common Cause:** Inappropriate parameters for data characteristics
- **Not a Problem:** Grid search tests multiple configurations

### Memory Issues

For large datasets:

- **Solution:** Reduce `MAX_SAMPLES` in evaluation script
- **Recommendation:** 50,000-100,000 samples sufficient for evaluation
- **Note:** Full dataset (1M samples) may require 8+ GB RAM

## References

1. Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*
3. Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale* (Prophet paper)
4. Statsmodels Documentation: https://www.statsmodels.org/
5. Prophet Documentation: https://facebook.github.io/prophet/

## Contact and Support

For issues or questions about statistical models implementation:
- Check residual diagnostic plots for model assumptions
- Compare results to ML baselines
- Consider hybrid approaches if neither model type excels alone

---

**Last Updated:** Task #79 - Statistical Models Implementation

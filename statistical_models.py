"""
Statistical Time Series Models for FSO Channel Power Estimation.

This module implements classical statistical forecasting models:
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMAX (Seasonal ARIMA with eXogenous variables)
- Prophet (Facebook's forecasting model)

These models work on univariate time series and provide baseline comparisons
to machine learning approaches.
"""

import time
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")


@dataclass
class StationarityTestResult:
    """Results from stationarity tests."""
    is_stationary: bool
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: Dict[str, float]
    kpss_statistic: float
    kpss_pvalue: float
    kpss_critical_values: Dict[str, float]
    recommended_differencing: int


def test_stationarity(
    data: np.ndarray,
    significance_level: float = 0.05,
    verbose: bool = True
) -> StationarityTestResult:
    """
    Test time series stationarity using ADF and KPSS tests.
    
    - ADF (Augmented Dickey-Fuller): H0 = non-stationary (reject if p < alpha)
    - KPSS: H0 = stationary (reject if p < alpha)
    
    Args:
        data: Time series data (1D array)
        significance_level: Significance level for tests (default 0.05)
        verbose: Print detailed results
        
    Returns:
        StationarityTestResult with test statistics and recommendations
    """
    # ADF Test (H0: non-stationary)
    adf_result = adfuller(data, autolag='AIC')
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_critical = {k: v for k, v in adf_result[4].items()}
    adf_stationary = adf_pvalue < significance_level
    
    # KPSS Test (H0: stationary)
    kpss_result = kpss(data, regression='c', nlags='auto')
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_critical = {k: v for k, v in kpss_result[3].items()}
    kpss_stationary = kpss_pvalue > significance_level
    
    # Both tests should agree for clear conclusion
    is_stationary = adf_stationary and kpss_stationary
    
    # Recommend differencing order
    if is_stationary:
        recommended_d = 0
    elif not adf_stationary:
        recommended_d = 1
    else:
        recommended_d = 1
    
    if verbose:
        print("\n" + "=" * 60)
        print("Stationarity Tests")
        print("=" * 60)
        print(f"\nADF Test (H0: non-stationary):")
        print(f"  Statistic: {adf_statistic:.4f}")
        print(f"  P-value: {adf_pvalue:.4f}")
        print(f"  Critical Values: {adf_critical}")
        print(f"  Conclusion: {'Stationary' if adf_stationary else 'Non-stationary'}")
        
        print(f"\nKPSS Test (H0: stationary):")
        print(f"  Statistic: {kpss_statistic:.4f}")
        print(f"  P-value: {kpss_pvalue:.4f}")
        print(f"  Critical Values: {kpss_critical}")
        print(f"  Conclusion: {'Stationary' if kpss_stationary else 'Non-stationary'}")
        
        print(f"\nFinal Assessment: {'Stationary' if is_stationary else 'Non-stationary'}")
        print(f"Recommended differencing order (d): {recommended_d}")
        print("=" * 60)
    
    return StationarityTestResult(
        is_stationary=is_stationary,
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        adf_critical_values=adf_critical,
        kpss_statistic=kpss_statistic,
        kpss_pvalue=kpss_pvalue,
        kpss_critical_values=kpss_critical,
        recommended_differencing=recommended_d
    )


@dataclass
class ModelResults:
    """Results from model training and evaluation."""
    model_name: str
    horizon: int
    train_time: float
    inference_time: float
    rmse: float
    mae: float
    variance_reduction: float
    predictions: np.ndarray
    actuals: np.ndarray
    residuals: np.ndarray
    best_params: Dict
    aic: Optional[float] = None
    bic: Optional[float] = None


class ARIMAForecaster:
    """ARIMA model for time series forecasting with automatic order selection."""
    
    def __init__(
        self,
        p_range: List[int] = [0, 1, 2, 5],
        d_range: List[int] = [0, 1, 2],
        q_range: List[int] = [0, 1, 2, 5],
        criterion: str = 'aic'
    ):
        """
        Initialize ARIMA forecaster.
        
        Args:
            p_range: AR order candidates
            d_range: Differencing order candidates
            q_range: MA order candidates
            criterion: Model selection criterion ('aic' or 'bic')
        """
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.criterion = criterion.lower()
        self.best_order = None
        self.model = None
        self.fitted_model = None
        
    def grid_search(
        self,
        train_data: np.ndarray,
        verbose: bool = True
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        Grid search over ARIMA orders to find best model.
        
        Args:
            train_data: Training time series
            verbose: Print progress
            
        Returns:
            Tuple of (best_order, best_criterion_value)
        """
        best_criterion = np.inf
        best_order = None
        results = []
        
        if verbose:
            print(f"\nPerforming ARIMA grid search...")
            print(f"Testing {len(self.p_range) * len(self.d_range) * len(self.q_range)} configurations...")
        
        for p in self.p_range:
            for d in self.d_range:
                for q in self.q_range:
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted = model.fit()
                        
                        criterion_value = fitted.aic if self.criterion == 'aic' else fitted.bic
                        results.append({
                            'order': (p, d, q),
                            'aic': fitted.aic,
                            'bic': fitted.bic
                        })
                        
                        if criterion_value < best_criterion:
                            best_criterion = criterion_value
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        # Skip configurations that fail to converge
                        continue
        
        if verbose and best_order:
            print(f"Best order: {best_order} with {self.criterion.upper()} = {best_criterion:.2f}")
            
            # Show top 5 models
            results_df = pd.DataFrame(results).sort_values('aic').head(5)
            print("\nTop 5 models by AIC:")
            print(results_df.to_string(index=False))
        
        return best_order, best_criterion
    
    def fit(
        self,
        train_data: np.ndarray,
        order: Optional[Tuple[int, int, int]] = None,
        auto_search: bool = True
    ):
        """
        Fit ARIMA model.
        
        Args:
            train_data: Training time series
            order: ARIMA order (p, d, q). If None, performs grid search
            auto_search: Perform grid search if order is None
        """
        if order is None and auto_search:
            order, _ = self.grid_search(train_data, verbose=True)
        elif order is None:
            order = (1, 0, 1)  # Default order
            
        self.best_order = order
        self.model = ARIMA(train_data, order=order)
        self.fitted_model = self.model.fit()
        
        return self
    
    def forecast(self, steps: int) -> np.ndarray:
        """
        Forecast multiple steps ahead.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def recursive_forecast(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        refit_interval: int = 100
    ) -> np.ndarray:
        """
        Recursive one-step-ahead forecasting.
        
        Args:
            train_data: Initial training data
            test_data: Test data for evaluation
            refit_interval: Refit model every N samples (0 = no refitting)
            
        Returns:
            Array of one-step-ahead predictions
        """
        predictions = []
        history = list(train_data)
        
        for i, true_val in enumerate(test_data):
            # Refit model periodically
            if refit_interval > 0 and i % refit_interval == 0:
                self.fit(np.array(history), order=self.best_order, auto_search=False)
            
            # One-step forecast
            pred = self.forecast(steps=1)[0]
            predictions.append(pred)
            
            # Update history with true value
            history.append(true_val)
        
        return np.array(predictions)


class SARIMAXForecaster:
    """SARIMAX model with seasonal components."""
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        exog_features: Optional[np.ndarray] = None
    ):
        """
        Initialize SARIMAX forecaster.
        
        Args:
            order: Non-seasonal ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            exog_features: Exogenous features (optional)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_features = exog_features
        self.model = None
        self.fitted_model = None
        
    def fit(self, train_data: np.ndarray):
        """Fit SARIMAX model."""
        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=self.exog_features
        )
        self.fitted_model = self.model.fit(disp=False)
        return self
    
    def forecast(self, steps: int, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast multiple steps ahead."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return np.array(forecast)
    
    def recursive_forecast(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        refit_interval: int = 100
    ) -> np.ndarray:
        """Recursive one-step-ahead forecasting."""
        predictions = []
        history = list(train_data)
        
        for i, true_val in enumerate(test_data):
            if refit_interval > 0 and i % refit_interval == 0:
                self.fit(np.array(history))
            
            pred = self.forecast(steps=1)[0]
            predictions.append(pred)
            history.append(true_val)
        
        return np.array(predictions)


class ProphetForecaster:
    """Prophet model for time series forecasting."""
    
    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive',
        changepoint_range: float = 0.8,
        daily_seasonality: bool = False,
        weekly_seasonality: bool = False,
        yearly_seasonality: bool = False
    ):
        """
        Initialize Prophet forecaster.
        
        Args:
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Flexibility of seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_range: Proportion of data for changepoint detection
            daily_seasonality: Enable daily seasonality
            weekly_seasonality: Enable weekly seasonality
            yearly_seasonality: Enable yearly seasonality
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
            
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.changepoint_range = changepoint_range
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None
        
    def fit(self, train_data: np.ndarray, freq: str = '100us'):
        """
        Fit Prophet model.
        
        Args:
            train_data: Training time series
            freq: Frequency string (e.g., '100us' for 10kHz)
        """
        # Create datetime index (Prophet requires datetime)
        dates = pd.date_range(start='2020-01-01', periods=len(train_data), freq=freq)
        df = pd.DataFrame({'ds': dates, 'y': train_data})
        
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            changepoint_range=self.changepoint_range,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality
        )
        
        # Suppress Prophet's verbose output
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        
        self.model.fit(df)
        return self
    
    def forecast(self, steps: int, freq: str = '100us') -> np.ndarray:
        """Forecast multiple steps ahead."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self.model.predict(future)
        
        # Return only the forecasted values (not historical fit)
        return forecast['yhat'].values[-steps:]


def recursive_multi_step_forecast(
    model,
    train_data: np.ndarray,
    test_data: np.ndarray,
    horizon: int,
    refit_interval: int = 100,
    verbose: bool = True
) -> np.ndarray:
    """
    Recursive multi-step forecasting for any statistical model.
    
    For each test point, forecast 'horizon' steps ahead and take the last prediction.
    This accumulates errors over the horizon.
    
    Args:
        model: Fitted statistical model (ARIMA, SARIMAX, or Prophet)
        train_data: Initial training data
        test_data: Test data
        horizon: Number of steps ahead to forecast
        refit_interval: Refit model every N samples
        verbose: Print progress
        
    Returns:
        Array of predictions for each test point
    """
    predictions = []
    history = list(train_data)
    
    if verbose:
        print(f"\nRecursive {horizon}-step ahead forecasting...")
        print(f"Test samples: {len(test_data)}, Refit interval: {refit_interval}")
    
    for i, true_val in enumerate(test_data):
        if verbose and (i % 1000 == 0 or i == 0):
            print(f"  Progress: {i}/{len(test_data)} samples...")
        
        # Refit model periodically
        if refit_interval > 0 and i % refit_interval == 0:
            if isinstance(model, ARIMAForecaster):
                model.fit(np.array(history), order=model.best_order, auto_search=False)
            elif isinstance(model, SARIMAXForecaster):
                model.fit(np.array(history))
            elif isinstance(model, ProphetForecaster):
                model.fit(np.array(history))
        
        # Multi-step forecast
        forecast = model.forecast(steps=horizon)
        pred = forecast[-1]  # Take the last (horizon-th) prediction
        predictions.append(pred)
        
        # Update history with true value (not prediction)
        history.append(true_val)
    
    return np.array(predictions)


def plot_residual_diagnostics(
    residuals: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Generate residual diagnostic plots.
    
    Args:
        residuals: Model residuals
        model_name: Name of model for title
        save_path: Path to save plot (if provided)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals over time
    axes[0, 0].plot(residuals, linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. ACF of residuals
    plot_acf(residuals, lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('ACF of Residuals')
    axes[1, 1].grid(alpha=0.3)
    
    fig.suptitle(f'{model_name} - Residual Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved residual diagnostics to {save_path}")
    
    plt.close()


def evaluate_statistical_model(
    model,
    model_name: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    horizon: int,
    refit_interval: int = 100
) -> ModelResults:
    """
    Evaluate a statistical model on test data.
    
    Args:
        model: Fitted statistical model
        model_name: Name for reporting
        train_data: Training data
        test_data: Test data
        horizon: Forecast horizon
        refit_interval: Refit interval for recursive forecasting
        
    Returns:
        ModelResults object with evaluation metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name} - Horizon {horizon}")
    print(f"{'=' * 60}")
    
    # Training time (already fitted, so record as 0)
    train_time = 0.0
    
    # Inference time
    start_time = time.time()
    predictions = recursive_multi_step_forecast(
        model, train_data, test_data, horizon, refit_interval, verbose=True
    )
    inference_time = time.time() - start_time
    
    # Calculate metrics
    residuals = test_data - predictions
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    
    # Variance reduction (compared to naive persistence forecast)
    naive_variance = np.var(test_data)
    residual_variance = np.var(residuals)
    variance_reduction = (naive_variance - residual_variance) / naive_variance
    
    # Extract model parameters
    best_params = {}
    if isinstance(model, ARIMAForecaster):
        best_params['order'] = model.best_order
        if model.fitted_model:
            best_params['aic'] = model.fitted_model.aic
            best_params['bic'] = model.fitted_model.bic
    elif isinstance(model, SARIMAXForecaster):
        best_params['order'] = model.order
        best_params['seasonal_order'] = model.seasonal_order
        if model.fitted_model:
            best_params['aic'] = model.fitted_model.aic
            best_params['bic'] = model.fitted_model.bic
    elif isinstance(model, ProphetForecaster):
        best_params['changepoint_prior_scale'] = model.changepoint_prior_scale
        best_params['seasonality_prior_scale'] = model.seasonality_prior_scale
    
    # Print results
    print(f"\n{'Results:':<20}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Variance Reduction: {variance_reduction:.2%}")
    print(f"  Inference Time: {inference_time:.2f}s ({inference_time/len(test_data)*1000:.2f}ms per sample)")
    
    return ModelResults(
        model_name=model_name,
        horizon=horizon,
        train_time=train_time,
        inference_time=inference_time,
        rmse=rmse,
        mae=mae,
        variance_reduction=variance_reduction,
        predictions=predictions,
        actuals=test_data,
        residuals=residuals,
        best_params=best_params,
        aic=best_params.get('aic'),
        bic=best_params.get('bic')
    )

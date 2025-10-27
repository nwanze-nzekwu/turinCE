"""
Gradient Boosting Models Module for FSO Channel Power Estimation.

This module implements XGBoost and LightGBM regression models with comprehensive
hyperparameter tuning, multi-horizon evaluation, and performance comparison.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

# Import gradient boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid


class GradientBoostingTrainer:
    """
    Trainer for gradient boosting models (XGBoost and LightGBM) with 
    hyperparameter tuning and comprehensive evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        
    def get_xgboost_param_grid(self) -> Dict:
        """
        Get hyperparameter grid for XGBoost tuning.
        
        Returns:
            Dictionary of hyperparameter options
        """
        return {
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'n_estimators': [100, 300, 500, 1000],
            'max_depth': [3, 5, 7, 10, None],
            'min_child_weight': [1, 5, 10, 20],
            'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
            'reg_lambda': [0, 0.1, 1.0],  # L2 regularization
            'colsample_bytree': [0.8, 0.9, 1.0],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    def get_lightgbm_param_grid(self) -> Dict:
        """
        Get hyperparameter grid for LightGBM tuning.
        
        Returns:
            Dictionary of hyperparameter options
        """
        return {
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'n_estimators': [100, 300, 500, 1000],
            'max_depth': [3, 5, 7, 10, -1],  # -1 means unlimited
            'min_data_in_leaf': [1, 5, 10, 20],
            'num_leaves': [31, 63, 127],
            'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
            'reg_lambda': [0, 0.1, 1.0],  # L2 regularization
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0]
        }
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None,
        use_gpu: bool = False
    ) -> Tuple[object, Dict]:
        """
        Train XGBoost model with given parameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: Model parameters (uses defaults if None)
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 5,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1.0,
            'colsample_bytree': 1.0,
            'subsample': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        # Add GPU support if requested
        if use_gpu:
            try:
                default_params['tree_method'] = 'gpu_hist'
                default_params['gpu_id'] = 0
            except Exception as e:
                warnings.warn(f"GPU not available, using CPU: {e}")
        
        # Train model with timing
        start_time = time.time()
        
        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = {
            'training_time': training_time,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'best_iteration': model.best_iteration,
            'n_estimators_used': model.n_estimators
        }
        
        return model, metrics
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ) -> Tuple[object, Dict]:
        """
        Train LightGBM model with given parameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: Model parameters (uses defaults if None)
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")
        
        # Default parameters
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 5,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'reg_alpha': 0,
            'reg_lambda': 1.0,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 5,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        # Train model with timing
        start_time = time.time()
        
        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = {
            'training_time': training_time,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'best_iteration': model.best_iteration_,
            'n_estimators_used': model.n_estimators
        }
        
        return model, metrics
    
    def tune_hyperparameters(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Optional[Dict] = None,
        max_trials: int = 50,
        use_gpu: bool = False
    ) -> Tuple[Dict, Dict]:
        """
        Perform hyperparameter tuning using grid/random search.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Parameter grid (uses default if None)
            max_trials: Maximum number of parameter combinations to try
            use_gpu: Whether to use GPU (XGBoost only)
            
        Returns:
            Tuple of (best parameters, all trial results)
        """
        if param_grid is None:
            if model_type == 'xgboost':
                param_grid = self.get_xgboost_param_grid()
            elif model_type == 'lightgbm':
                param_grid = self.get_lightgbm_param_grid()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Sample random combinations if grid is too large
        all_combinations = list(ParameterGrid(param_grid))
        if len(all_combinations) > max_trials:
            np.random.seed(self.random_state)
            combinations = [all_combinations[i] for i in 
                          np.random.choice(len(all_combinations), max_trials, replace=False)]
        else:
            combinations = all_combinations
        
        print(f"\nTuning {model_type} with {len(combinations)} parameter combinations...")
        
        best_score = float('inf')
        best_params = None
        all_results = []
        
        for i, params in enumerate(combinations):
            if i % 10 == 0:
                print(f"  Trial {i+1}/{len(combinations)}...")
            
            try:
                if model_type == 'xgboost':
                    model, metrics = self.train_xgboost(
                        X_train, y_train, X_val, y_val, params, use_gpu
                    )
                else:
                    model, metrics = self.train_lightgbm(
                        X_train, y_train, X_val, y_val, params
                    )
                
                val_score = metrics['val_rmse']
                
                result = {
                    'params': params.copy(),
                    'val_rmse': val_score,
                    'train_rmse': metrics['train_rmse'],
                    'training_time': metrics['training_time']
                }
                all_results.append(result)
                
                if val_score < best_score:
                    best_score = val_score
                    best_params = params.copy()
                    print(f"  New best RMSE: {best_score:.6f}")
                    
            except Exception as e:
                print(f"  Trial {i+1} failed: {e}")
                continue
        
        print(f"\nBest validation RMSE: {best_score:.6f}")
        print(f"Best parameters: {best_params}")
        
        return best_params, all_results
    
    def evaluate_model(
        self,
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
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
            # Pre-compensated = actual - predicted (error)
            precomp_variance = np.var(y_train - train_pred)
            metrics['precomp_variance_train'] = precomp_variance
        
        return metrics
    
    def get_feature_importance(
        self,
        model: object,
        feature_names: List[str],
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            importance_type: Type of importance ('gain', 'split', 'cover' for XGBoost;
                           'split' or 'gain' for LightGBM)
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if isinstance(model, xgb.XGBRegressor):
            importance = model.get_booster().get_score(importance_type=importance_type)
            # Convert feature names (f0, f1, ...) to actual names
            importance_dict = {}
            for key, value in importance.items():
                if key.startswith('f'):
                    idx = int(key[1:])
                    if idx < len(feature_names):
                        importance_dict[feature_names[idx]] = value
                else:
                    importance_dict[key] = value
        elif isinstance(model, lgb.LGBMRegressor):
            importance = model.booster_.feature_importance(importance_type=importance_type)
            importance_dict = dict(zip(feature_names, importance))
        else:
            raise ValueError("Model type not supported for feature importance extraction")
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df


def train_and_evaluate_horizons(
    datasets: Dict,
    horizons: List[int],
    model_type: str = 'xgboost',
    tune_params: bool = True,
    max_tuning_trials: int = 50,
    use_gpu: bool = False,
    random_state: int = 42
) -> Dict:
    """
    Train and evaluate gradient boosting models across multiple horizons.
    
    Args:
        datasets: Dictionary of datasets from prepare_dataset()
        horizons: List of prediction horizons to evaluate
        model_type: 'xgboost' or 'lightgbm'
        tune_params: Whether to perform hyperparameter tuning
        max_tuning_trials: Maximum tuning trials per horizon
        use_gpu: Whether to use GPU (XGBoost only)
        random_state: Random seed
        
    Returns:
        Dictionary of results for each horizon
    """
    trainer = GradientBoostingTrainer(random_state=random_state)
    results = {}
    
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"Processing horizon: {horizon} samples")
        print(f"{'='*80}")
        
        if horizon not in datasets:
            print(f"Warning: Horizon {horizon} not found in datasets. Skipping...")
            continue
        
        # Get data for this horizon
        X_train, y_train = datasets[horizon]['train']
        X_val, y_val = datasets[horizon]['val']
        X_test, y_test = datasets[horizon]['test']
        feature_names = datasets[horizon]['feature_names']
        
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
        print(f"Number of features: {len(feature_names)}")
        
        # Hyperparameter tuning if requested
        best_params = None
        if tune_params:
            best_params, tuning_results = trainer.tune_hyperparameters(
                model_type, X_train, y_train, X_val, y_val,
                max_trials=max_tuning_trials, use_gpu=use_gpu
            )
        
        # Train final model with best parameters
        print(f"\nTraining final {model_type} model...")
        if model_type == 'xgboost':
            model, train_metrics = trainer.train_xgboost(
                X_train, y_train, X_val, y_val, best_params, use_gpu
            )
        else:
            model, train_metrics = trainer.train_lightgbm(
                X_train, y_train, X_val, y_val, best_params
            )
        
        print(f"  Training RMSE: {train_metrics['train_rmse']:.6f}")
        print(f"  Validation RMSE: {train_metrics['val_rmse']:.6f}")
        print(f"  Training time: {train_metrics['training_time']:.2f}s")
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        test_metrics = trainer.evaluate_model(model, X_test, y_test, X_train, y_train)
        print(f"  Test RMSE: {test_metrics['test_rmse']:.6f}")
        print(f"  Test MAE: {test_metrics['test_mae']:.6f}")
        print(f"  Inference time per sample: {test_metrics['inference_time_per_sample_ms']:.4f}ms")
        
        # Get feature importance
        feature_importance = trainer.get_feature_importance(model, feature_names)
        
        # Store results
        results[horizon] = {
            'model': model,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'feature_names': feature_names
        }
        
        # Display top features
        print(f"\nTop 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Gradient Boosting Models Module")
    print("Import this module to use GradientBoostingTrainer class")
    print("\nExample:")
    print("  from gradient_boosting_models import train_and_evaluate_horizons")
    print("  from data_preparation import prepare_dataset")
    print("  ")
    print("  datasets = prepare_dataset('strong')")
    print("  results = train_and_evaluate_horizons(datasets, [50, 100, 200, 500], 'xgboost')")

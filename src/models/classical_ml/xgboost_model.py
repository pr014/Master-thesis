"""XGBoost model wrapper for ECG LOS regression."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class XGBoostECGModel:
    """
    XGBoost model wrapper for LOS regression.
    
    Wraps xgboost.XGBRegressor with convenience methods for training,
    evaluation, and model persistence.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        random_state: int = 42,
    ):
        """
        Initialize XGBoost model.
        
        Args:
            config: Configuration dictionary with XGBoost hyperparameters.
            random_state: Random seed for reproducibility.
        """
        xgb_config = config.get("xgboost", {})
        
        # In XGBoost 2.x, early_stopping_rounds should be set during initialization
        # but only if a validation set will be provided. We set it to None initially
        # and will update it in fit() if a validation set is provided
        self.early_stopping_rounds = xgb_config.get("early_stopping_rounds", 20)
        
        self.model = xgb.XGBRegressor(
            n_estimators=xgb_config.get("n_estimators", 200),
            max_depth=xgb_config.get("max_depth", 6),
            learning_rate=xgb_config.get("learning_rate", 0.1),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            random_state=random_state,
            n_jobs=xgb_config.get("n_jobs", -1),
            tree_method=xgb_config.get("tree_method", "hist"),
            verbosity=xgb_config.get("verbosity", 1),
            early_stopping_rounds=None,  # Set to None initially, will be set in fit() if needed
        )
        
        self.config = config
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "XGBoostECGModel":
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features of shape (N, feature_dim).
            y_train: Training labels of shape (N,).
            X_val: Validation features of shape (M, feature_dim) (optional).
            y_val: Validation labels of shape (M,) (optional).
            verbose: Whether to print training progress.
        
        Returns:
            self for method chaining.
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            # For XGBoost 2.x, set early_stopping_rounds on the model if validation set is provided
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        else:
            # Disable early stopping if no validation set
            self.model.set_params(early_stopping_rounds=None)
        
        # In XGBoost 2.x, early_stopping_rounds is set during model initialization
        # and will be used automatically when eval_set is provided
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose,
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LOS values.
        
        Args:
            X: Features of shape (N, feature_dim).
        
        Returns:
            Predictions of shape (N,).
        """
        return self.model.predict(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model and compute metrics.
        
        Args:
            X: Features of shape (N, feature_dim).
            y: True labels of shape (N,).
        
        Returns:
            Dictionary with metrics:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r2: R² Score
            - median_ae: Median Absolute Error
        """
        y_pred = self.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Median Absolute Error
        absolute_errors = np.abs(y_pred - y)
        median_ae = np.median(absolute_errors)
        
        # Percentile errors
        p25_error = np.percentile(absolute_errors, 25)
        p50_error = np.percentile(absolute_errors, 50)
        p75_error = np.percentile(absolute_errors, 75)
        p90_error = np.percentile(absolute_errors, 90)
        
        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "median_ae": float(median_ae),
            "p25_error": float(p25_error),
            "p50_error": float(p50_error),
            "p75_error": float(p75_error),
            "p90_error": float(p90_error),
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array of shape (feature_dim,).
        """
        return self.model.feature_importances_
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model (.pkl or .json).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == ".json":
            self.model.save_model(str(filepath))
        else:
            # Default: use pickle
            import pickle
            with open(filepath, "wb") as f:
                pickle.dump(self.model, f)
    
    def load_model(self, filepath: str) -> "XGBoostECGModel":
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model (.pkl or .json).
        
        Returns:
            self for method chaining.
        """
        filepath = Path(filepath)
        
        if filepath.suffix == ".json":
            self.model.load_model(str(filepath))
        else:
            # Default: use pickle
            import pickle
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)
        
        return self


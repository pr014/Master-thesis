"""XGBoost model wrapper for ECG LOS regression."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
        self.mortality_model = None  # Optional XGBClassifier, set when multi_task.enabled
    
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
    
    def fit_mortality(
        self,
        X_train: np.ndarray,
        y_mortality: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val_mortality: Optional[np.ndarray] = None,
        scale_pos_weight: Optional[float] = None,
        verbose: bool = True,
    ) -> "XGBoostECGModel":
        """Train mortality classifier (binary). Call after fit() if multi_task enabled."""
        cfg = self.config.get("xgboost_mortality", self.config.get("xgboost", {}))
        scale = scale_pos_weight
        if scale is None and cfg.get("scale_pos_weight") == "auto" and y_mortality is not None:
            n_pos = int(np.sum(y_mortality == 1))
            n_neg = int(np.sum(y_mortality == 0))
            scale = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0
        self.mortality_model = xgb.XGBClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 4),
            learning_rate=cfg.get("learning_rate", 0.1),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            random_state=self.config.get("seed", 42),
            n_jobs=cfg.get("n_jobs", -1),
            tree_method=cfg.get("tree_method", "hist"),
            verbosity=cfg.get("verbosity", 1),
            early_stopping_rounds=cfg.get("early_stopping_rounds", 10),
            scale_pos_weight=scale,
        )
        eval_set = None
        if X_val is not None and y_val_mortality is not None:
            eval_set = [(X_train, y_mortality), (X_val, y_val_mortality)]
        self.mortality_model.fit(X_train, y_mortality, eval_set=eval_set, verbose=verbose)
        return self

    def predict_mortality_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict mortality probabilities. Returns (N,) proba of class 1."""
        if self.mortality_model is None:
            raise RuntimeError("Mortality model not fitted. Enable multi_task and call fit_mortality.")
        return self.mortality_model.predict_proba(X)[:, 1]

    def evaluate_mortality(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate mortality classifier. y: 0/1 labels. Filters out invalid (e.g. -1)."""
        valid = (y >= 0) & (y <= 1)
        if not np.any(valid):
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}
        X_v, y_v = X[valid], y[valid].astype(int)
        proba = self.predict_mortality_proba(X_v)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_v, pred)
        prec = precision_score(y_v, pred, zero_division=0)
        rec = recall_score(y_v, pred, zero_division=0)
        f1 = f1_score(y_v, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_v, proba) if len(np.unique(y_v)) > 1 else 0.0
        except Exception:
            auc = 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

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
            import pickle
            payload = {"los_model": self.model}
            if self.mortality_model is not None:
                payload["mortality_model"] = self.mortality_model
            with open(filepath, "wb") as f:
                pickle.dump(payload, f)
    
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
            import pickle
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.model = data["los_model"]
                self.mortality_model = data.get("mortality_model")
            else:
                self.model = data  # legacy: only LOS model
                self.mortality_model = None
        
        return self


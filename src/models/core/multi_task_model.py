"""Multi-Task ECG Model wrapper for LOS regression + Mortality prediction."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from .base_model import BaseECGModel
from ...data.ecg.ecg_dataset import ehr_window_feature_dim


class MultiTaskECGModel(nn.Module):
    """Wrapper for existing ECG models to add mortality prediction head.
    
    This wrapper takes a base model (e.g., CNNScratch, HybridCNNLSTM) and adds
    a second output head for mortality prediction while keeping the original
    LOS head (now for regression).
    
    Architecture:
    - Shared Backbone: Base model without final FC layer
    - LOS Head: FC layer for regression (output dim = 1)
    - Mortality Head: FC layer for binary classification (0/1)
    """
    
    def __init__(self, base_model: BaseECGModel, config: Dict[str, Any]):
        """Initialize Multi-Task model.
        
        Args:
            base_model: Base ECG model (must implement get_features() method).
            config: Configuration dictionary. Should contain:
                - training.dropout_rate: Dropout rate for mortality head
                - multi_task: Optional multi-task specific config
                - data.task_type: "regression" (default) or "classification"
        """
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Get task type
        data_config = config.get("data", {})
        self.task_type = data_config.get("task_type", "regression")
        
        # Get dropout rate from config
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get feature dimension by running a dummy forward pass
        # This works for models that implement get_features()
        # Set model to eval mode to avoid BatchNorm issues with single sample
        was_training = base_model.training
        base_model.eval()
        
        # Get device from model parameters (to ensure dummy_input is on same device)
        device = next(base_model.parameters()).device
        dummy_input = torch.zeros(1, 12, 5000, device=device)
        
        # Check if demographic features are enabled
        demographic_config = data_config.get("demographic_features", {})
        use_demographics = demographic_config.get("enabled", False)
        dummy_demographic_features = None
        if use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3
            dummy_demographic_features = torch.zeros(1, demo_dim, device=device)
        
        # Check if ICU unit features are enabled
        icu_unit_config = data_config.get("icu_unit_features", {})
        use_icu_units = icu_unit_config.get("enabled", False)
        icu_unit_list = icu_unit_config.get("icu_unit_list", [])
        icu_unit_dim = len(icu_unit_list) if use_icu_units else 0
        dummy_icu_unit_features = None
        if use_icu_units:
            dummy_icu_unit_features = torch.zeros(1, icu_unit_dim, device=device)
        
        sofa_config = data_config.get("sofa_features", {})
        use_sofa = sofa_config.get("enabled", False)
        sofa_columns = sofa_config.get("columns", ["sofa_total"])
        if isinstance(sofa_columns, str):
            sofa_columns = [sofa_columns]
        sofa_dim = len(sofa_columns) if use_sofa else 0
        dummy_sofa_features = None
        if use_sofa and sofa_dim > 0:
            dummy_sofa_features = torch.zeros(1, sofa_dim, device=device)
        self._pass_sofa_to_backbone = bool(use_sofa and sofa_dim > 0)

        therapy_config = data_config.get("icu_therapy_support_features", {})
        use_therapy = therapy_config.get("enabled", False)
        therapy_columns = therapy_config.get(
            "columns",
            [
                "mech_vent",
                "niv_hfnc",
                "vaso_any",
                "vaso_non_catechol_any",
                "rrt",
            ],
        )
        if isinstance(therapy_columns, str):
            therapy_columns = [therapy_columns]
        therapy_dim = len(therapy_columns) if use_therapy else 0
        dummy_icu_therapy_features = None
        if use_therapy and therapy_dim > 0:
            dummy_icu_therapy_features = torch.zeros(1, therapy_dim, device=device)
        self._pass_icu_therapy_to_backbone = bool(use_therapy and therapy_dim > 0)

        ehr_cfg = data_config.get("ehr_window_features", {})
        use_ehr = ehr_cfg.get("enabled", False)
        ehr_dim = ehr_window_feature_dim(ehr_cfg) if use_ehr else 0
        dummy_ehr_features = None
        if use_ehr and ehr_dim > 0:
            dummy_ehr_features = torch.zeros(1, ehr_dim, device=device)
        self._pass_ehr_window_to_backbone = bool(use_ehr and ehr_dim > 0)
        
        with torch.no_grad():
            try:
                _gf_kw = dict(
                    demographic_features=dummy_demographic_features,
                    icu_unit_features=dummy_icu_unit_features,
                )
                if self._pass_sofa_to_backbone:
                    _gf_kw["sofa_features"] = dummy_sofa_features
                if self._pass_icu_therapy_to_backbone:
                    _gf_kw["icu_therapy_support_features"] = dummy_icu_therapy_features
                if self._pass_ehr_window_to_backbone:
                    _gf_kw["ehr_window_features"] = dummy_ehr_features
                features = base_model.get_features(dummy_input, **_gf_kw)
                feature_dim = features.shape[1]
            except NotImplementedError:
                raise NotImplementedError(
                    f"{base_model.__class__.__name__} does not support multi-task learning. "
                    "The model must implement get_features() method."
                )
        # Restore original training mode
        if was_training:
            base_model.train()
        
        # Create LOS head for regression (output dim = 1)
        self.los_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1),
        )
        
        # Create mortality head
        # Architecture: Dropout -> Linear -> Sigmoid
        self.mortality_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # Output probability 0.0-1.0
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        demographic_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None,
        sofa_features: Optional[torch.Tensor] = None,
        icu_therapy_support_features: Optional[torch.Tensor] = None,
        ehr_window_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-task model.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
            icu_unit_features: Optional tensor of shape (B, icu_unit_dim) containing one-hot ICU unit.
                              None if ICU unit features are disabled.
        
        Returns:
            Dictionary with:
                - 'los': LOS regression output of shape (B, 1) - continuous LOS in days
                - 'mortality': Mortality probabilities of shape (B, 1) in range [0, 1]
        """
        # Extract features using base model (includes demographic and ICU unit features if enabled)
        _gf_kw = dict(
            demographic_features=demographic_features,
            icu_unit_features=icu_unit_features,
        )
        if getattr(self, "_pass_sofa_to_backbone", False):
            _gf_kw["sofa_features"] = sofa_features
        if getattr(self, "_pass_icu_therapy_to_backbone", False):
            _gf_kw["icu_therapy_support_features"] = icu_therapy_support_features
        if getattr(self, "_pass_ehr_window_to_backbone", False):
            _gf_kw["ehr_window_features"] = ehr_window_features
        features = self.base_model.get_features(x, **_gf_kw)  # (B, feature_dim)
        
        # LOS regression head
        los_predictions = self.los_head(features)  # (B, 1) - continuous LOS in days
        
        # Mortality prediction head
        mortality_probs = self.mortality_head(features)  # (B, 1)
        
        return {
            "los": los_predictions,
            "mortality": mortality_probs
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from both tasks.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
        
        Returns:
            Dictionary with:
                - 'los': LOS predictions of shape (B,) - continuous values in days
                - 'mortality': Mortality binary predictions of shape (B,) (0 or 1)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            los_predictions = outputs["los"].squeeze(-1)  # (B,) - continuous LOS
            mortality_predictions = (outputs["mortality"] > 0.5).long().squeeze(-1)
        
        return {
            "los": los_predictions,
            "mortality": mortality_predictions
        }
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters.
        
        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

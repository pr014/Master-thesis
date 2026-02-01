"""ECG-CPC model with S4 backbone and CPC pretraining.

Based on Al-Masud et al. (2025): "Benchmarking ECG Foundational Models"
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from scipy import signal as scipy_signal
import pickle
import io

from ..core.base_model import BaseECGModel
from .s4_encoder import S4Encoder
from .rnn_encoder import RNNEncoder


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that skips missing modules (e.g., clinical_ts)."""
    
    def __init__(self, file, *args, **kwargs):
        super().__init__(file, *args, **kwargs)
        # Handle persistent IDs (PyTorch Lightning uses these for tensor storage)
        self.persistent_load = self._persistent_load
    
    def _persistent_load(self, pid):
        """Handle persistent IDs used by PyTorch for tensor storage."""
        # Return None for persistent IDs - PyTorch will handle tensor loading separately
        return None
    
    def find_class(self, module, name):
        """Override to handle missing modules gracefully."""
        # Skip problematic modules by returning a dummy class
        if any(skip in module for skip in ['clinical_ts', 'omegaconf', 'hydra']):
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __getattr__(self, name):
                    return DummyClass()
                def __setattr__(self, name, value):
                    object.__setattr__(self, name, value)
                def __call__(self, *args, **kwargs):
                    return DummyClass()
            return DummyClass
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Fallback: return dummy class for any missing module
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __getattr__(self, name):
                    return DummyClass()
                def __setattr__(self, name, value):
                    object.__setattr__(self, name, value)
            return DummyClass


class ECG_S4_CPC(BaseECGModel):
    """ECG-CPC model with RNN Encoder + S4 backbone for multi-task learning.
    
    Architecture (matching checkpoint for Transfer Learning):
    - Input: (B, 12, 5000) ECG + (B, 2) Demographics
    - Input Resampling: (B, 12, 5000) -> (B, 12, 600) [10s @ 500Hz -> 2.5s @ 240Hz]
    - RNN Encoder: 4 layers, features [512, 512, 512, 512], strides [2, 1, 1, 1]
    - S4 Encoder: 4 layers, d_model=512, d_state=8
    - Global Average Pooling: (B, seq_len, 512) -> (B, 512)
    - Late Fusion: Concat([ECG(512), age(1), sex(1)]) -> (B, 514)
    - Shared Layer: BatchNorm -> Dropout -> Linear(514->128) -> ReLU -> Dropout
    - LOS Head: Linear(128->10) -> (B, 10) logits
    - Mortality Head: Linear(128->1) -> (B, 1) probs
    
    Based on Al-Masud et al. (2025) and Gu et al. (2022).
    Matches checkpoint architecture for full Transfer Learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ECG-CPC model.
        
        Args:
            config: Configuration dictionary containing:
                - model.num_classes: Number of LOS classes (default: 10)
                - model.d_model: S4 hidden dimension (default: 256)
                - model.d_state: S4 state dimension (default: 64)
                - model.n_layers: Number of S4 layers (default: 4)
                - model.dropout: S4 dropout rate (default: 0.1)
                - model.prenorm: Use pre-normalization (default: True)
                - model.pretrained.enabled: Whether to load pretrained weights
                - model.pretrained.weights_path: Path to pretrained weights
                - training.dropout_rate: Dropout for shared layer (default: 0.3)
        """
        super().__init__(config)
        
        # Get training config
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get model config
        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", self.num_classes)
        self.num_classes = num_classes
        
        # RNN Encoder hyperparameters (from checkpoint)
        rnn_features = model_config.get("rnn_features", [512, 512, 512, 512])
        rnn_kernel_sizes = model_config.get("rnn_kernel_sizes", [3, 1, 1, 1])
        rnn_strides = model_config.get("rnn_strides", [2, 1, 1, 1])
        rnn_dropout = model_config.get("rnn_dropout", 0.1)
        
        # S4 Encoder hyperparameters (from checkpoint: model_dim=512, state_dim=8)
        d_model = model_config.get("d_model", 512)  # Changed from 256 to 512
        d_state = model_config.get("d_state", 8)    # Changed from 64 to 8
        n_layers = model_config.get("n_layers", 4)
        s4_dropout = model_config.get("dropout", 0.1)
        prenorm = model_config.get("prenorm", True)
        
        # Input handling: Resample from 5000 (10s @ 500Hz) to 600 (2.5s @ 240Hz)
        # Checkpoint expects: 2.5s @ 240Hz = 600 timesteps
        # Our data: 10s @ 500Hz = 5000 timesteps
        self.target_seq_len = model_config.get("target_seq_len", 600)  # For checkpoint compatibility
        self.use_input_resampling = model_config.get("use_input_resampling", True)
        
        # Sampling rates for frequency-based resampling
        data_config = config.get("data", {})
        self.original_sampling_rate = data_config.get("sampling_rate", 500.0)  # 500 Hz
        # Calculate target sampling rate: 600 samples / 2.5s = 240 Hz
        input_seconds = model_config.get("input_seconds", 2.5)  # 2.5 seconds for checkpoint
        self.target_sampling_rate = self.target_seq_len / input_seconds  # 240.0 Hz
        
        # Input: (B, 12, 5000) -> Resample to (B, 12, 600) if needed
        d_input = 12  # Number of ECG leads
        
        # RNN Encoder: (B, 600, 12) -> (B, ~300, 512) after first stride=2
        self.rnn_encoder = RNNEncoder(
            d_input=d_input,
            features=rnn_features,
            kernel_sizes=rnn_kernel_sizes,
            strides=rnn_strides,
            dropout=rnn_dropout,
        )
        
        # S4 Encoder: (B, seq_len_after_rnn, 512) -> (B, seq_len_after_rnn, 512)
        # Input to S4 is the output from RNN (512 dim)
        self.s4_encoder = S4Encoder(
            d_input=rnn_features[-1],  # 512 (from RNN output)
            d_model=d_model,           # 512
            d_state=d_state,           # 8
            n_layers=n_layers,
            dropout=s4_dropout,
            prenorm=prenorm,
        )
        
        # Global Average Pooling: (B, 5000, 256) -> (B, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Will be applied after transpose
        
        # Check if demographic features are enabled
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        use_demographics = demographic_config.get("enabled", False)
        
        # Feature dimension after pooling
        feature_dim = d_model  # 512 (changed from 256)
        
        # Late fusion with demographics
        if use_demographics:
            demo_dim = 2  # Age + Sex
            feature_dim += demo_dim  # 514 (changed from 258)
        self.use_demographics = use_demographics
        
        # Shared layer: BatchNorm -> Dropout -> Linear -> ReLU -> Dropout
        self.shared_bn = nn.BatchNorm1d(feature_dim)
        self.shared_dropout1 = nn.Dropout(dropout_rate)
        self.shared_fc = nn.Linear(feature_dim, 128)
        self.shared_relu = nn.ReLU()
        self.shared_dropout2 = nn.Dropout(dropout_rate)
        
        # Task-specific heads
        # LOS Head: (B, 128) -> (B, 10) logits
        self.los_head = nn.Linear(128, num_classes)
        
        # Mortality Head: (B, 128) -> (B, 1) probabilities
        self.mortality_head = nn.Linear(128, 1)
        
        # Load pretrained weights if enabled
        self._load_pretrained_weights(config)
    
    def freeze_backbone(self) -> None:
        """Freeze RNN and S4 encoders for Phase 1 training (frozen backbone)."""
        for param in self.rnn_encoder.parameters():
            param.requires_grad = False
        for param in self.s4_encoder.parameters():
            param.requires_grad = False
        print("RNN and S4 encoders frozen (Phase 1: frozen backbone)")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze RNN and S4 encoders for Phase 2 training (fine-tuning)."""
        for param in self.rnn_encoder.parameters():
            param.requires_grad = True
        for param in self.s4_encoder.parameters():
            param.requires_grad = True
        print("RNN and S4 encoders unfrozen (Phase 2: fine-tuning)")
    
    def _resample_input(self, x: torch.Tensor) -> torch.Tensor:
        """Frequency-based resampling from 500Hz to 240Hz, then crop to 2.5s.
        
        This performs proper frequency-domain resampling (not just linear interpolation)
        to match the checkpoint's expected input format.
        
        Process:
        1. Resample from 500Hz → 240Hz: (B, 12, 5000) @ 500Hz → (B, 12, 2400) @ 240Hz
        2. Crop to 2.5s: (B, 12, 2400) → (B, 12, 600) [first 2.5s @ 240Hz]
        
        Args:
            x: Input tensor of shape (B, 12, 5000) - 10s @ 500Hz
            
        Returns:
            Resampled tensor of shape (B, 12, 600) - 2.5s @ 240Hz
        """
        if not self.use_input_resampling:
            return x
        
        B, C, T = x.shape
        
        if T == self.target_seq_len and self.original_sampling_rate == self.target_sampling_rate:
            return x
        
        # Step 1: Frequency-based resampling from 500Hz → 240Hz
        # Calculate number of samples after resampling: 10s @ 240Hz = 2400 samples
        num_samples_240hz = int(T * self.target_sampling_rate / self.original_sampling_rate)
        # num_samples_240hz = int(5000 * 240 / 500) = 2400
        
        # Convert to numpy for scipy.signal.resample (FFT-based resampling)
        device = x.device
        dtype = x.dtype
        x_np = x.detach().cpu().numpy()  # (B, 12, 5000)
        
        # Resample each sample in batch
        x_resampled_list = []
        for b in range(B):
            # Resample each sample: (12, 5000) @ 500Hz → (12, 2400) @ 240Hz
            x_b = x_np[b]  # (12, 5000)
            # scipy.signal.resample uses FFT-based resampling (frequency-domain)
            # axis=1 means resample along the time dimension
            x_b_resampled = scipy_signal.resample(x_b, num_samples_240hz, axis=1)  # (12, 2400)
            
            # Step 2: Crop to 2.5s (first 600 samples @ 240Hz)
            x_b_cropped = x_b_resampled[:, :self.target_seq_len]  # (12, 600)
            x_resampled_list.append(x_b_cropped)
        
        # Stack back to batch and convert to torch
        x_resampled = np.stack(x_resampled_list, axis=0)  # (B, 12, 600)
        x_resampled = torch.from_numpy(x_resampled).to(device=device, dtype=dtype)
        
        return x_resampled  # (B, 12, 600) @ 240Hz
    
    def _load_pretrained_weights(self, config: Dict[str, Any]) -> None:
        """Load pretrained ECG-CPC weights (RNN + S4).
        
        Args:
            config: Configuration dictionary with pretrained settings.
        """
        pretrained_config = config.get("model", {}).get("pretrained", {})
        enabled = pretrained_config.get("enabled", False)
        
        if not enabled:
            print("Pretrained weights disabled. Training from scratch.")
            return
        
        weights_path = pretrained_config.get("weights_path", "")
        if not weights_path:
            print("Warning: pretrained.enabled is True but weights_path is empty. Training from scratch.")
            return
        
        # Resolve path
        weights_path = Path(weights_path)
        if not weights_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent.parent
            weights_path = project_root / weights_path
        
        if not weights_path.exists():
            print(f"Warning: Pretrained weights file not found at {weights_path}. Training from scratch.")
            return
        
        try:
            print(f"Loading pretrained ECG-CPC weights from: {weights_path}")
            # Create mock modules for missing dependencies (clinical_ts, etc.)
            import sys
            import types
            
            # Create a mock class factory
            class MockClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __getattr__(self, name):
                    return MockClass()
                def __call__(self, *args, **kwargs):
                    return MockClass()
            
            # Mock missing packages and submodules BEFORE torch.load
            # This must be done BEFORE torch.load, as torch.load uses pickle internally
            
            # Create a function to dynamically create mock modules and classes
            def create_mock_module(module_name):
                """Create a mock module that returns MockClass for any attribute."""
                if module_name not in sys.modules:
                    # Create a proper module type
                    mock_module = types.ModuleType(module_name)
                    sys.modules[module_name] = mock_module
            
            # Create proper mock classes that inherit from object (required for pickle)
            class MockDictConfig(dict):
                """Mock DictConfig class that inherits from dict (required for pickle)."""
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    if args:
                        if isinstance(args[0], dict):
                            self.update(args[0])
                def __getattr__(self, name):
                    return None
                def __setattr__(self, name, value):
                    if name.startswith('_'):
                        super().__setattr__(name, value)
                    else:
                        self[name] = value
            
            class MockContainerMetadata:
                """Mock ContainerMetadata class."""
                def __init__(self, *args, **kwargs):
                    pass
                def __getattr__(self, name):
                    return None
            
            # Mock clinical_ts and all its submodules
            for module_name in [
                'clinical_ts',
                'clinical_ts.ts',
                'clinical_ts.template_modules',
                'clinical_ts.models',
                'clinical_ts.data',
                'clinical_ts.utils',
            ]:
                create_mock_module(module_name)
            
            # Mock omegaconf and all its submodules
            for module_name in [
                'omegaconf',
                'omegaconf.dictconfig',
                'omegaconf.base',
                'omegaconf.listconfig',
            ]:
                create_mock_module(module_name)
            
            # Add DictConfig class to omegaconf.dictconfig (must be a real class, not a MockClass)
            if 'omegaconf.dictconfig' in sys.modules:
                sys.modules['omegaconf.dictconfig'].DictConfig = MockDictConfig
            
            # Add DictConfig and ContainerMetadata to omegaconf.base
            if 'omegaconf.base' in sys.modules:
                sys.modules['omegaconf.base'].DictConfig = MockDictConfig
                sys.modules['omegaconf.base'].ContainerMetadata = MockContainerMetadata
            
            # Try to load checkpoint
            try:
                # First try with weights_only=True (safer, PyTorch 2.0+)
                try:
                    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
                except Exception:
                    # Fallback: Use weights_only=False (will use mocked modules)
                    # The mocked modules should handle missing clinical_ts.ts
                    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}")
            
            # Handle different checkpoint formats (PyTorch Lightning)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Separate RNN (ts.enc) and S4 (ts.pred) weights
            rnn_keys = {}
            s4_keys = {}
            other_keys = []
            
            for key, value in state_dict.items():
                if 'ts.enc' in key or 'enc' in key.lower():
                    # RNN encoder weights
                    # Remove prefix: "ts.enc." or "model.ts.enc."
                    clean_key = key
                    for prefix in ["model.ts.enc.", "ts.enc.", "model.", "backbone.", "module."]:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    rnn_keys[clean_key] = value
                elif 'ts.pred' in key or 'pred' in key.lower() or 's4' in key.lower():
                    # S4 predictor weights
                    # Remove prefix: "ts.pred." or "model.ts.pred."
                    clean_key = key
                    for prefix in ["model.ts.pred.", "ts.pred.", "model.", "backbone.", "module."]:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    s4_keys[clean_key] = value
                else:
                    other_keys.append(key)
            
            # Load RNN encoder weights
            rnn_state_dict = self.rnn_encoder.state_dict()
            rnn_pretrained = {}
            rnn_skipped = []
            
            # Map checkpoint RNN keys to our RNN encoder structure
            # Checkpoint uses Conv1D layers, we need to match layer indices
            for our_key in rnn_state_dict.keys():
                # Our keys: "layers.0.weight", "layers.1.weight", etc.
                # Checkpoint might use different naming
                found = False
                for ckpt_key, ckpt_value in rnn_keys.items():
                    # Try to match by layer index and parameter type
                    our_layer_idx = our_key.split('.')[1] if '.' in our_key else None
                    our_param = our_key.split('.')[-1]  # "weight", "bias", etc.
                    
                    # Check if shapes match
                    if rnn_state_dict[our_key].shape == ckpt_value.shape:
                        # Additional check: try to match by layer index
                        if our_layer_idx:
                            # Check if checkpoint key contains similar layer index
                            if our_layer_idx in ckpt_key or (not found):
                                rnn_pretrained[our_key] = ckpt_value
                                found = True
                                break
                        else:
                            rnn_pretrained[our_key] = ckpt_value
                            found = True
                            break
                
                if not found:
                    rnn_skipped.append(our_key)
            
            # Load S4 encoder weights
            s4_state_dict = self.s4_encoder.state_dict()
            s4_pretrained = {}
            s4_skipped = []
            
            # Map checkpoint S4 keys to our S4 encoder structure
            for our_key in s4_state_dict.keys():
                found = False
                for ckpt_key, ckpt_value in s4_keys.items():
                    # Try to match by parameter name and shape
                    our_param = our_key.split('.')[-1]  # "weight", "bias", etc.
                    
                    if s4_state_dict[our_key].shape == ckpt_value.shape:
                        # Try to match by parameter name
                        if our_param in ckpt_key.lower() or ckpt_key.lower().endswith(our_param):
                            s4_pretrained[our_key] = ckpt_value
                            found = True
                            break
                        elif not found:
                            # Fallback: shape match
                            s4_pretrained[our_key] = ckpt_value
                            found = True
                            break
                
                if not found:
                    s4_skipped.append(our_key)
            
            # Load weights
            total_loaded = 0
            if rnn_pretrained:
                self.rnn_encoder.load_state_dict(rnn_pretrained, strict=False)
                print(f"Loaded {len(rnn_pretrained)} pretrained RNN encoder layers.")
                total_loaded += len(rnn_pretrained)
                if rnn_skipped:
                    print(f"  Skipped {len(rnn_skipped)} RNN layers (not matching).")
            
            if s4_pretrained:
                self.s4_encoder.load_state_dict(s4_pretrained, strict=False)
                print(f"Loaded {len(s4_pretrained)} pretrained S4 encoder layers.")
                total_loaded += len(s4_pretrained)
                if s4_skipped:
                    print(f"  Skipped {len(s4_skipped)} S4 layers (not matching).")
            
            if total_loaded == 0:
                print("Warning: No matching pretrained weights found. Training from scratch.")
                print(f"  Available keys in checkpoint: {list(cleaned_state_dict.keys())[:10]}...")
            else:
                print(f"Successfully loaded {total_loaded} pretrained layers total.")
                
        except Exception as e:
            print(f"Error loading pretrained weights: {e}. Training from scratch.")
            import traceback
            traceback.print_exc()
    
    def _forward_features(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through feature extraction (before heads).
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000)
            demographic_features: Optional demographic features of shape (B, 2)
            
        Returns:
            Features tensor of shape (B, 128)
        """
        # Resample input: (B, 12, 5000) -> (B, 12, 600) for checkpoint compatibility
        x = self._resample_input(x)  # (B, 12, 600)
        
        # Transpose: (B, 12, 600) -> (B, 600, 12) for RNN
        x = x.transpose(1, 2)  # (B, 600, 12)
        
        # RNN Encoder: (B, 600, 12) -> (B, ~300, 512) after first stride=2
        x = self.rnn_encoder(x)  # (B, seq_len_after_rnn, 512)
        
        # S4 Encoder: (B, seq_len_after_rnn, 512) -> (B, seq_len_after_rnn, 512)
        x = self.s4_encoder(x)  # (B, seq_len_after_rnn, 512)
        
        # Transpose back for pooling: (B, seq_len_after_rnn, 512) -> (B, 512, seq_len_after_rnn)
        x = x.transpose(1, 2)  # (B, 512, seq_len_after_rnn)
        
        # Global Average Pooling: (B, 512, seq_len_after_rnn) -> (B, 512, 1)
        x = self.global_pool(x)  # (B, 512, 1)
        
        # Squeeze: (B, 512, 1) -> (B, 512)
        x = x.squeeze(-1)  # (B, 512)
        
        # Late fusion with demographics
        if self.use_demographics and demographic_features is not None:
            # Concatenate: (B, 512) + (B, 2) -> (B, 514)
            x = torch.cat([x, demographic_features], dim=1)
        
        # Shared layer
        # BatchNorm expects (B, C) for 1D
        if x.dim() == 2:
            x = self.shared_bn(x)  # (B, 514) or (B, 512)
        x = self.shared_dropout1(x)
        x = self.shared_fc(x)  # (B, 128)
        x = self.shared_relu(x)
        x = self.shared_dropout2(x)
        
        return x  # (B, 128)
    
    def forward(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning LOS logits and mortality probabilities.
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000) - will be resampled to (B, 12, 600)
            demographic_features: Optional demographic features of shape (B, 2)
            
        Returns:
            Tuple of:
                - los_logits: LOS classification logits of shape (B, num_classes)
                - mortality_probs: Mortality probabilities of shape (B, 1)
        """
        # Extract features
        features = self._forward_features(x, demographic_features=demographic_features)
        
        # Task-specific heads
        los_logits = self.los_head(features)  # (B, 10)
        mortality_logits = self.mortality_head(features)  # (B, 1)
        mortality_probs = torch.sigmoid(mortality_logits)  # (B, 1)
        
        return los_logits, mortality_probs
    
    def get_features(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features before task-specific heads.
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000) - will be resampled to (B, 12, 600)
            demographic_features: Optional demographic features of shape (B, 2)
            
        Returns:
            Features tensor of shape (B, 128)
        """
        return self._forward_features(x, demographic_features=demographic_features)


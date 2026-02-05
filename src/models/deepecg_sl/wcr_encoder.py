"""WCR Transformer Encoder Wrapper for DeepECG-SL."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from pathlib import Path

from .checkpoint_utils import load_wcr_checkpoint
from .huggingface_loader import HuggingFaceModelLoader


class WCREncoder(nn.Module):
    """WCR Transformer Encoder wrapper.
    
    Loads pretrained base_ssl.pt model and extracts only the encoder
    (without task-specific head).
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize WCR Encoder.
        
        Args:
            config: Configuration dictionary. Should contain:
                - model.wcr.model_name: Model name (e.g., "wcr_77_classes")
                - model.wcr.huggingface_repo: HuggingFace repo ID
                - model.wcr.base_ssl_path: Path to base_ssl.pt (relative to model dir)
                - model.pretrained.cache_dir: Cache directory for models
            device: Device to load model on (optional)
        """
        super().__init__()
        
        model_config = config.get("model", {})
        wcr_config = model_config.get("wcr", {})
        pretrained_config = model_config.get("pretrained", {})
        
        model_name = wcr_config.get("model_name", "wcr_77_classes")
        cache_dir = pretrained_config.get("cache_dir", "data/pretrained_weights/deepecg_sl")
        base_ssl_path = wcr_config.get("base_ssl_path", "base_ssl.pt")
        
        # Resolve cache directory path (relative to project root)
        cache_dir_path = Path(cache_dir)
        if not cache_dir_path.is_absolute():
            # Try to find project root (go up from src/models/deepecg_sl/wcr_encoder.py)
            # wcr_encoder.py -> deepecg_sl -> models -> src -> MA-thesis-1 (project root)
            project_root = Path(__file__).parent.parent.parent.parent
            cache_dir_path = project_root / cache_dir
        
        # Handle base_ssl_path - try absolute first, then relative to cache_dir
        base_ssl_path_obj = Path(base_ssl_path)
        if base_ssl_path_obj.is_absolute():
            # If absolute path, check if it exists
            if base_ssl_path_obj.exists():
                base_ssl_direct = base_ssl_path_obj
            else:
                # Absolute path doesn't exist, use just the filename in cache_dir
                base_ssl_direct = cache_dir_path / base_ssl_path_obj.name
        else:
            # Relative path, use it relative to cache_dir
            base_ssl_direct = cache_dir_path / base_ssl_path
        
        # Checkpoint is always relative to cache_dir
        checkpoint_direct = cache_dir_path / f"{model_name}.pt"
        
        if base_ssl_direct.exists() and checkpoint_direct.exists():
            # Files are directly in cache_dir - use them
            print(f"Found weights directly in cache directory: {cache_dir_path}")
            model_dir = cache_dir_path
            checkpoint_path = str(checkpoint_direct)
            base_ssl_full_path = str(base_ssl_direct)
        else:
            # Try old structure: cache_dir/model_name/
            model_dir_path = cache_dir_path / model_name
            base_ssl_old = model_dir_path / base_ssl_path
            
            if base_ssl_old.exists():
                # Old structure found
                print(f"Found weights in subdirectory: {model_dir_path}")
                model_dir = model_dir_path
                # Find checkpoint file (any .pt file except base_ssl.pt)
                checkpoint_files = [f for f in model_dir.glob("*.pt") if f.name != "base_ssl.pt"]
                if not checkpoint_files:
                    raise FileNotFoundError(
                        f"No checkpoint file found in {model_dir} (excluding base_ssl.pt)"
                    )
                checkpoint_path = str(checkpoint_files[0])
                base_ssl_full_path = str(base_ssl_old)
            else:
                # Try to download from HuggingFace (fallback)
                print(f"Weights not found locally. Attempting download from HuggingFace...")
                model_dir = HuggingFaceModelLoader.load_wcr_model(model_name, str(cache_dir_path))
                model_dir = Path(model_dir)
                # Find checkpoint file (any .pt file except base_ssl.pt)
                checkpoint_files = [f for f in model_dir.glob("*.pt") if f.name != "base_ssl.pt"]
                if not checkpoint_files:
                    raise FileNotFoundError(
                        f"No checkpoint file found in {model_dir} (excluding base_ssl.pt)"
                    )
                checkpoint_path = str(checkpoint_files[0])
                base_ssl_full_path = str(model_dir / base_ssl_path)
        
        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = str(device)
        
        # Load model using fairseq-signals
        print(f"Loading WCR checkpoint from: {checkpoint_path}")
        print(f"Using base SSL model: {base_ssl_full_path}")
        
        model, args, task = load_wcr_checkpoint(
            checkpoint_path=checkpoint_path,
            base_ssl_path=base_ssl_full_path,
            map_location=map_location
        )
        
        # Extract encoder from model
        # The model structure depends on fairseq-signals, typically:
        # model.encoder is the transformer encoder
        if hasattr(model, "encoder"):
            self.encoder = model.encoder
            self._use_full_model = False
        elif hasattr(model, "model") and hasattr(model.model, "encoder"):
            self.encoder = model.model.encoder
            self._use_full_model = False
        else:
            # If encoder is not directly accessible, use the whole model
            # but we'll need to extract features differently
            self.encoder = model
            self._use_full_model = True
        
        self.model = model
        self.args = args
        self.task = task
        self.device = device
        
        print(f"WCR Encoder loaded on {device}")
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through WCR encoder.
        
        Args:
            x: Input tensor of shape (B, 12, 2500) or (B, 2500, 12)
            padding_mask: Optional padding mask tensor
            
        Returns:
            Encoder output of shape (B, seq_len, d_model)
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        
        # fairseq-signals expects (B, channels, seq_len) format = (B, 12, seq_len)
        # The ConvFeatureExtraction uses nn.Conv1d which expects (B, C, L)
        # NO transpose needed - keep input as (B, 12, seq_len)
        
        # Create net_input dictionary
        net_input = {"source": x}
        if padding_mask is not None:
            net_input["padding_mask"] = padding_mask
        
        # Forward through encoder
        # Always use the full model approach to ensure compatibility with fairseq-signals
        # The model handles padding_mask correctly via net_input dictionary
        net_output = self.model(**net_input)
        
        # Extract encoder output from net_output
        if hasattr(net_output, "encoder_out"):
            encoder_out = net_output.encoder_out
        elif isinstance(net_output, dict) and "encoder_out" in net_output:
            encoder_out = net_output["encoder_out"]
        elif hasattr(net_output, "encoder_states") and net_output.encoder_states:
            # Some fairseq models return encoder_states as a list
            encoder_out = net_output.encoder_states[-1]  # Take the last layer
        else:
            # Last resort: try to extract from encoder directly if accessible
            # But don't pass padding_mask as keyword argument
            if hasattr(self, "encoder") and self.encoder is not None:
                try:
                    # Try calling encoder without padding_mask
                    encoder_out = self.encoder(x)
                except (TypeError, AttributeError):
                    # If that fails, raise a more informative error
                    raise RuntimeError(
                        f"Could not extract encoder output from model. "
                        f"Model output type: {type(net_output)}, "
                        f"Available attributes: {dir(net_output) if hasattr(net_output, '__dict__') else 'N/A'}"
                    )
            else:
                raise RuntimeError(
                    f"Could not extract encoder output from model. "
                    f"Model output type: {type(net_output)}"
                )
        
        return encoder_out
    
    def freeze(self) -> None:
        """Freeze encoder parameters (no gradient updates)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()  # Set to eval mode
    
    def unfreeze(self) -> None:
        """Unfreeze encoder parameters (enable gradient updates)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()  # Set to train mode


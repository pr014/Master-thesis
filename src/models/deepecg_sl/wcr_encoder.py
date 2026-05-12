"""WCR Transformer Encoder Wrapper for DeepECG-SL."""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path

from .checkpoint_utils import load_wcr_checkpoint


def _find_wcr_pretrained_pair(
    project_root: Path,
    cache_dir_path: Path,
    model_name: str,
    base_ssl_name: str,
) -> Optional[Tuple[Path, Path]]:
    """Locate (wcr_checkpoint.pt, base_ssl.pt) for common on-disk layouts."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        p = p.resolve()
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        candidates.append(p)

    _add(cache_dir_path)
    # Under the same cache_dir (e.g. .../deepecg_sl/), Linux is case-sensitive: WCR_77_classes vs wcr_77_classes.
    _add(cache_dir_path / "WCR_77_classes")
    _add(cache_dir_path / "wcr_77_classes")
    pw = project_root / "data" / "pretrained_weights"
    dsl = pw / "deepecg_sl"
    _add(dsl / "WCR_77_classes")
    _add(dsl / "wcr_77_classes")
    _add(pw / "WCR_77_classes")
    _add(pw / "wcr_77_classes")
    _add(pw / model_name)
    if model_name.lower() == "wcr_77_classes":
        _add(pw / "WCR_77_classes")
    # Flat layout: base_ssl.pt + <model>.pt next to each other
    _add(pw)

    for root in candidates:
        if not root.is_dir():
            continue
        base_ssl = root / base_ssl_name
        if not base_ssl.is_file():
            continue
        flat_ckpt = root / f"{model_name}.pt"
        if flat_ckpt.is_file() and flat_ckpt.name != base_ssl_name:
            return flat_ckpt.resolve(), base_ssl.resolve()
        pts = sorted(
            f
            for f in root.glob("*.pt")
            if f.is_file() and f.name != base_ssl_name
        )
        if pts:
            return pts[0].resolve(), base_ssl.resolve()
    return None


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
                - model.wcr.model_name: Subdirectory under cache_dir (e.g. "WCR_77_classes")
                - model.wcr.base_ssl_path: Filename or path for base_ssl.pt
                - model.pretrained.cache_dir: Base directory for local weights
            device: Device to load model on (optional)
        """
        super().__init__()
        
        model_config = config.get("model", {})
        wcr_config = model_config.get("wcr", {})
        pretrained_config = model_config.get("pretrained", {})
        
        model_name = wcr_config.get("model_name", "wcr_77_classes")
        cache_dir = pretrained_config.get("cache_dir", "data/pretrained_weights/deepecg_sl")
        base_ssl_path = wcr_config.get("base_ssl_path", "base_ssl.pt")
        # Basename only for paths under cache_dir/model_name (avoid join with wrongly expanded absolute path)
        base_ssl_name = Path(base_ssl_path).name
        
        project_root = Path(__file__).parent.parent.parent.parent
        # Resolve cache directory path (relative to project root)
        cache_dir_path = Path(cache_dir)
        if not cache_dir_path.is_absolute():
            # wcr_encoder.py -> deepecg_sl -> models -> src -> MA-thesis-1 (project root)
            cache_dir_path = (project_root / cache_dir).resolve()
        else:
            cache_dir_path = cache_dir_path.resolve()

        # Some checkpoints store a non-portable cache_dir (e.g. "/deepecg_sl") → /deepecg_sl/base_ssl.pt on Linux.
        # If that tree has no weights, fall back to the repo layout: data/pretrained_weights/deepecg_sl/
        def _weights_layout_ok(cdir: Path) -> bool:
            if (cdir / base_ssl_name).is_file() and (cdir / f"{model_name}.pt").is_file():
                return True
            for sub in (model_name, "WCR_77_classes", "wcr_77_classes"):
                b = cdir / sub / base_ssl_name
                if not b.is_file():
                    continue
                d = cdir / sub
                if any(
                    f.is_file() and f.name != base_ssl_name for f in d.glob("*.pt")
                ):
                    return True
            return False

        default_deepecg_sl = (project_root / "data" / "pretrained_weights" / "deepecg_sl").resolve()
        if not _weights_layout_ok(cache_dir_path) and _weights_layout_ok(default_deepecg_sl):
            print(
                f"[WCR] cache_dir from config not usable ({cache_dir!r} -> {cache_dir_path}); "
                f"using {default_deepecg_sl}"
            )
            cache_dir_path = default_deepecg_sl

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
            base_ssl_direct = cache_dir_path / base_ssl_name
        
        # Checkpoint is always relative to cache_dir
        checkpoint_direct = cache_dir_path / f"{model_name}.pt"
        
        if base_ssl_direct.exists() and checkpoint_direct.exists():
            # Files are directly in cache_dir - use them
            print(f"Found weights directly in cache directory: {cache_dir_path}")
            model_dir = cache_dir_path
            checkpoint_path = str(checkpoint_direct)
            base_ssl_full_path = str(base_ssl_direct)
        else:
            # Subdirectory layout: cache_dir/<subdir>/base_ssl.pt + other .pt
            # Prefer config model_name first (unchanged behaviour); then case variants for Linux (APFS often is not).
            model_dir_path: Optional[Path] = None
            base_ssl_old: Optional[Path] = None
            for sub in (model_name, "WCR_77_classes", "wcr_77_classes"):
                md = cache_dir_path / sub
                cand = md / base_ssl_name
                if cand.is_file():
                    model_dir_path = md
                    base_ssl_old = cand
                    break

            if model_dir_path is not None and base_ssl_old is not None:
                print(f"Found weights in subdirectory: {model_dir_path}")
                model_dir = model_dir_path
                checkpoint_files = [
                    f for f in model_dir_path.glob("*.pt") if f.is_file() and f.name != base_ssl_name
                ]
                if not checkpoint_files:
                    raise FileNotFoundError(
                        f"No checkpoint file found in {model_dir} (excluding {base_ssl_name!r})"
                    )
                checkpoint_path = str(checkpoint_files[0])
                base_ssl_full_path = str(base_ssl_old.resolve())
            else:
                found = _find_wcr_pretrained_pair(
                    project_root, cache_dir_path, model_name, base_ssl_name
                )
                if found is not None:
                    ck, bs = found
                    checkpoint_path = str(ck)
                    base_ssl_full_path = str(bs)
                    model_dir = ck.parent
                    print(f"Found WCR weights via fallback search under: {model_dir}")
                else:
                    flat_ckpt = cache_dir_path / f"{model_name}.pt"
                    subdir = cache_dir_path / model_name
                    raise FileNotFoundError(
                        "WCR weights not found locally (no HuggingFace download). "
                        "Expected either:\n"
                        f"  1) {base_ssl_direct} and {flat_ckpt}, or\n"
                        f"  2) {subdir / base_ssl_name} plus a second .pt checkpoint in {subdir}, or\n"
                        f"  3) the same under cache_dir/WCR_77_classes/ (case variant)\n"
                        f"(model.wcr.model_name={model_name!r}, cache_dir={cache_dir_path})"
                    )
        
        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = str(device)
        
        # Ensure absolute paths (fairseq resolves relative paths from cwd)
        checkpoint_path = str(Path(checkpoint_path).resolve())
        base_ssl_full_path = str(Path(base_ssl_full_path).resolve())
        
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


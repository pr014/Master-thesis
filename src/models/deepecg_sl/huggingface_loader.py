"""HuggingFace Model Loader for WCR pretrained weights."""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download


class HuggingFaceModelLoader:
    """Loader for WCR models from HuggingFace with automatic download and caching."""
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get HuggingFace API key from environment variable.
        
        Returns:
            API key string or None if not set
        """
        return os.getenv("HUGGINGFACE_API_KEY")
    
    @staticmethod
    def load_wcr_model(model_name: str, cache_dir: str) -> str:
        """Load WCR model from HuggingFace or local cache.
        
        Args:
            model_name: Model name (e.g., "wcr_77_classes")
            cache_dir: Base cache directory for models
            
        Returns:
            Path to model directory containing base_ssl.pt and task weights
            
        Raises:
            ValueError: If HUGGINGFACE_API_KEY is not set
            FileNotFoundError: If model files are not found after download
        """
        # Construct paths
        repo_id = f"heartwise/{model_name}"
        local_dir = Path(cache_dir) / model_name
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists locally
        base_ssl_path = local_dir / "base_ssl.pt"
        if base_ssl_path.exists():
            # Check if there are other .pt files (task weights)
            pt_files = list(local_dir.glob("*.pt"))
            if len(pt_files) > 1:  # base_ssl.pt + at least one task weight
                print(f"Model {model_name} already exists in {local_dir}")
                return str(local_dir)
        
        # Model not found locally, need to download from HuggingFace
        # Get API key (only needed for download)
        api_key = HuggingFaceModelLoader.get_api_key()
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY environment variable required to download model. "
                "Set it with: export HUGGINGFACE_API_KEY='your_key'\n"
                f"Or ensure model {model_name} is already in {local_dir}"
            )
        
        # Download from HuggingFace
        print(f"Downloading {repo_id} from HuggingFace...")
        print(f"Cache directory: {local_dir}")
        
        try:
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                repo_type="model",
                token=api_key
            )
            print(f"Successfully downloaded {repo_id} to {downloaded_path}")
            
            # Verify that base_ssl.pt exists
            base_ssl_path = Path(downloaded_path) / "base_ssl.pt"
            if not base_ssl_path.exists():
                raise FileNotFoundError(
                    f"base_ssl.pt not found in downloaded model at {downloaded_path}. "
                    "The model structure may be incorrect."
                )
            
            return downloaded_path
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model {repo_id} from HuggingFace: {e}\n"
                "Please check:\n"
                "1. HUGGINGFACE_API_KEY is set correctly\n"
                "2. You have access to the heartwise models repository\n"
                "3. Internet connection is available"
            ) from e


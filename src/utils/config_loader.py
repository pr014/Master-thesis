"""YAML configuration loader with hierarchical merging and environment variable support."""

from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file.
    
    Returns:
        dict: Configuration dictionary.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.
    
    The override dictionary takes precedence over the base dictionary.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.
    
    Returns:
        dict: Merged configuration dictionary.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def resolve_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path, supporting environment variables and relative/absolute paths.
    
    Priority:
    1. Environment variable if path starts with '$'
    2. Absolute path (if already absolute)
    3. Relative to base_dir (if provided) or current working directory
    
    Args:
        path: Path string (may contain environment variables like $HOME or $DATA_DIR).
        base_dir: Base directory for resolving relative paths.
    
    Returns:
        Path: Resolved Path object.
    """
    # Handle environment variables (e.g., $DATA_DIR/path/to/file)
    if path.startswith('$'):
        # Extract env var name and rest of path
        parts = path[1:].split('/', 1)
        env_var = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
        
        env_value = os.getenv(env_var)
        if env_value is None:
            raise ValueError(f"Environment variable {env_var} not set")
        
        path = os.path.join(env_value, rest) if rest else env_value
    
    path_obj = Path(path)
    
    # If absolute, return as-is
    if path_obj.is_absolute():
        return path_obj
    
    # If relative, resolve against base_dir or current working directory
    if base_dir is not None:
        return (Path(base_dir) / path_obj).resolve()
    
    return Path(path).resolve()


def expand_paths_in_config(config: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Recursively expand paths in configuration dictionary.
    
    Expands paths in string values that look like file/directory paths.
    Supports environment variables (e.g., $DATA_DIR/path).
    
    Args:
        config: Configuration dictionary.
        base_dir: Base directory for resolving relative paths.
    
    Returns:
        dict: Configuration with expanded paths.
    """
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = expand_paths_in_config(value, base_dir)
        elif isinstance(value, str) and key.endswith(('_dir', '_path', '_file', 'dir', 'path')):
            # Try to resolve as path
            try:
                result[key] = str(resolve_path(value, base_dir))
            except (ValueError, OSError):
                # If resolution fails, keep original value
                result[key] = value
        else:
            result[key] = value
    
    return result


def load_config(
    base_config_path: Optional[Path] = None,
    model_config_path: Optional[Path] = None,
    experiment_config_path: Optional[Path] = None,
    expand_paths: bool = True,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load and merge configuration files hierarchically.
    
    Configuration hierarchy (lowest to highest priority):
    1. base_config_path (if provided)
    2. model_config_path (if provided)
    3. experiment_config_path (if provided)
    4. Environment variables (for paths)
    
    If only model_config_path is provided (and base_config_path is None),
    the model config is loaded as a standalone complete configuration.
    This supports the new config structure where each model config contains
    all parameters (data, preprocessing, augmentation, training, model).
    
    Args:
        base_config_path: Path to base configuration (optional, no default).
        model_config_path: Path to model-specific configuration (can be standalone).
        experiment_config_path: Path to experiment-specific configuration.
        expand_paths: Whether to expand paths using environment variables (default: True).
        base_dir: Base directory for resolving relative paths (default: current working directory).
    
    Returns:
        dict: Merged configuration dictionary with expanded paths.
    
    Raises:
        ValueError: If neither base_config_path nor model_config_path is provided.
    """
    # Use project root as base_dir if not specified
    if base_dir is None:
        # Try to find project root (directory containing configs/)
        current = Path.cwd()
        while current != current.parent:
            if (current / "configs").exists():
                base_dir = current
                break
            current = current.parent
        else:
            base_dir = Path.cwd()
    
    # Initialize config
    config = {}
    
    # Load base config if provided
    if base_config_path is not None:
        config = load_yaml(base_config_path)
    
    # Load and merge model config
    if model_config_path is not None:
        model_config = load_yaml(model_config_path)
        if config:
            # Merge with base config
            config = merge_configs(config, model_config)
        else:
            # Use model config as standalone complete config
            config = model_config
    
    # Ensure we have some configuration
    if not config:
        raise ValueError(
            "At least one of base_config_path or model_config_path must be provided"
        )
    
    # Merge experiment config if provided
    if experiment_config_path is not None:
        experiment_config = load_yaml(experiment_config_path)
        config = merge_configs(config, experiment_config)
    
    # Expand paths if requested
    if expand_paths:
        config = expand_paths_in_config(config, base_dir)
    
    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary to save.
        output_path: Path where to save the configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

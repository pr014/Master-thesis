"""ECG data augmentation transforms."""

from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
import numpy as np


class GaussianNoise(nn.Module):
    """Add Gaussian noise to ECG signal.
    
    Adds random noise from N(0, σ²) where σ = noise_std × signal_std.
    """
    
    def __init__(self, noise_std: float = 0.03):
        """Initialize Gaussian noise augmentation.
        
        Args:
            noise_std: Noise standard deviation as fraction of signal std (default: 0.03).
        """
        super().__init__()
        self.noise_std = noise_std
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise.
        
        Args:
            signal: ECG signal tensor (C, T) or (B, C, T).
        
        Returns:
            Augmented signal with same shape.
        """
        if not self.training:
            return signal
        
        # Check for NaN/Inf in input
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            return signal  # Return unchanged if input is invalid
        
        # Compute signal std per lead (channel)
        if signal.dim() == 2:  # (C, T)
            signal_std = signal.std(dim=1, keepdim=True)
        else:  # (B, C, T)
            signal_std = signal.std(dim=2, keepdim=True)
        
        # Replace zero or NaN std with small value to avoid NaN
        signal_std = torch.clamp(signal_std, min=1e-6)
        signal_std = torch.where(torch.isnan(signal_std), torch.tensor(1e-6, device=signal.device), signal_std)
        
        # Generate noise
        noise = torch.randn_like(signal) * self.noise_std * signal_std
        result = signal + noise
        
        # Check for NaN/Inf in output
        if torch.isnan(result).any() or torch.isinf(result).any():
            return signal  # Return original if augmentation produced NaN
        
        return result


class AmplitudeScaling(nn.Module):
    """Scale ECG signal amplitude by random factor.
    
    Multiplies signal by random factor α ∈ [scale_min, scale_max].
    """
    
    def __init__(self, scale_min: float = 0.85, scale_max: float = 1.15):
        """Initialize amplitude scaling augmentation.
        
        Args:
            scale_min: Minimum scaling factor (default: 0.85).
            scale_max: Maximum scaling factor (default: 1.15).
        """
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply amplitude scaling.
        
        Args:
            signal: ECG signal tensor (C, T) or (B, C, T).
        
        Returns:
            Scaled signal with same shape.
        """
        if not self.training:
            return signal
        
        # Check for NaN/Inf in input
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            return signal  # Return unchanged if input is invalid
        
        # Generate random scale factor per lead (channel)
        if signal.dim() == 2:  # (C, T)
            num_leads = signal.shape[0]
            scale = torch.empty(num_leads, 1, device=signal.device).uniform_(
                self.scale_min, self.scale_max
            )
        else:  # (B, C, T)
            batch_size, num_leads = signal.shape[0], signal.shape[1]
            scale = torch.empty(batch_size, num_leads, 1, device=signal.device).uniform_(
                self.scale_min, self.scale_max
            )
        
        result = signal * scale
        
        # Check for NaN/Inf in output
        if torch.isnan(result).any() or torch.isinf(result).any():
            return signal  # Return original if augmentation produced NaN
        
        return result


class BaselineWander(nn.Module):
    """Add low-frequency sinusoidal baseline wander to ECG signal.
    
    Adds sinusoidal drift with frequency f ∈ [freq_min, freq_max] Hz
    and amplitude A ∈ [amp_min, amp_max] mV.
    """
    
    def __init__(
        self,
        freq_min: float = 0.15,
        freq_max: float = 0.4,
        amp_min: float = 0.03,
        amp_max: float = 0.08,
        sampling_rate: float = 500.0,
    ):
        """Initialize baseline wander augmentation.
        
        Args:
            freq_min: Minimum frequency in Hz (default: 0.15).
            freq_max: Maximum frequency in Hz (default: 0.4).
            amp_min: Minimum amplitude in mV (default: 0.03).
            amp_max: Maximum amplitude in mV (default: 0.08).
            sampling_rate: ECG sampling rate in Hz (default: 500.0).
        """
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.amp_min = amp_min
        self.amp_max = amp_max
        self.sampling_rate = sampling_rate
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply baseline wander.
        
        Args:
            signal: ECG signal tensor (C, T) or (B, C, T).
        
        Returns:
            Signal with baseline wander added.
        """
        if not self.training:
            return signal
        
        # Check for NaN/Inf in input
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            return signal  # Return unchanged if input is invalid
        
        # Get signal length
        if signal.dim() == 2:  # (C, T)
            num_samples = signal.shape[1]
            num_leads = signal.shape[0]
            batch_size = 1
        else:  # (B, C, T)
            batch_size, num_leads, num_samples = signal.shape
        
        # Create time vector
        t = torch.arange(num_samples, device=signal.device, dtype=torch.float32) / self.sampling_rate
        
        # Generate random frequency and amplitude per lead
        if signal.dim() == 2:
            freq = torch.empty(num_leads, 1, device=signal.device).uniform_(self.freq_min, self.freq_max)
            amp = torch.empty(num_leads, 1, device=signal.device).uniform_(self.amp_min, self.amp_max)
            phase = torch.empty(num_leads, 1, device=signal.device).uniform_(0, 2 * np.pi)
        else:
            freq = torch.empty(batch_size, num_leads, 1, device=signal.device).uniform_(
                self.freq_min, self.freq_max
            )
            amp = torch.empty(batch_size, num_leads, 1, device=signal.device).uniform_(
                self.amp_min, self.amp_max
            )
            phase = torch.empty(batch_size, num_leads, 1, device=signal.device).uniform_(0, 2 * np.pi)
        
        # Generate sinusoidal baseline wander
        if signal.dim() == 2:
            # (num_leads, 1) * (num_samples,) -> (num_leads, num_samples)
            baseline = amp * torch.sin(2 * np.pi * freq * t.unsqueeze(0) + phase)
        else:
            # (batch_size, num_leads, 1) * (num_samples,) -> (batch_size, num_leads, num_samples)
            baseline = amp * torch.sin(2 * np.pi * freq * t.unsqueeze(0).unsqueeze(0) + phase)
        
        result = signal + baseline
        
        # Check for NaN/Inf in output
        if torch.isnan(result).any() or torch.isinf(result).any():
            return signal  # Return original if augmentation produced NaN
        
        return result


class ECGAugmentation(nn.Module):
    """Composable ECG augmentation transform.
    
    Applies multiple augmentation techniques in sequence.
    Only active during training (when model.train() is called).
    
    Works with preprocessed .npy data:
    - Input: torch.Tensor (C, T) where C=12 leads, T=5000 samples (10s @ 500Hz)
    - Data should already be normalized and filtered
    - Augmentation is applied after loading from .npy files
    """
    
    def __init__(
        self,
        gaussian_noise: bool = False,
        noise_std: float = 0.03,
        amplitude_scaling: bool = False,
        scale_min: float = 0.85,
        scale_max: float = 1.15,
        sampling_rate: float = 500.0,
    ):
        """Initialize ECG augmentation.
        
        Args:
            gaussian_noise: Enable Gaussian noise augmentation.
            noise_std: Noise std as fraction of signal std.
            amplitude_scaling: Enable amplitude scaling.
            scale_min: Minimum scaling factor.
            scale_max: Maximum scaling factor.
            sampling_rate: ECG sampling rate (Hz).
        """
        super().__init__()
        
        self.transforms = nn.ModuleList()
        
        if gaussian_noise:
            self.transforms.append(GaussianNoise(noise_std=noise_std))
        
        if amplitude_scaling:
            self.transforms.append(AmplitudeScaling(scale_min=scale_min, scale_max=scale_max))
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply augmentations sequentially.
        
        Args:
            signal: ECG signal tensor (C, T) or (B, C, T).
        
        Returns:
            Augmented signal.
        """
        if not self.training:
            return signal
        
        x = signal
        for transform in self.transforms:
            x = transform(x)
        return x


def create_augmentation_transform(config: Dict[str, Any]) -> Optional[Callable]:
    """Create augmentation transform from config.
    
    Args:
        config: Configuration dictionary with augmentation settings.
    
    Returns:
        Augmentation transform or None if disabled.
    """
    aug_config = config.get("data", {}).get("augmentation", {})
    
    if not aug_config.get("enabled", False):
        return None
    
    return ECGAugmentation(
        gaussian_noise=aug_config.get("gaussian_noise", False),
        noise_std=aug_config.get("noise_std", 0.03),
        amplitude_scaling=aug_config.get("amplitude_scaling", False),
        scale_min=aug_config.get("scale_min", 0.85),
        scale_max=aug_config.get("scale_max", 1.15),
        sampling_rate=config.get("data", {}).get("sampling_rate", 500.0),
    )


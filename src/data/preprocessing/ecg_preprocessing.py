"""ECG preprocessing pipeline: Filtering → Resampling → Segmentation → Normalization.

Pipeline order:
1. Bandpass filtering (0.5-50 Hz, Butterworth 4th order) with filtfilt
2. Notch filtering (60 Hz for US power line frequency) to remove power line interference
3. Resampling to target sampling rate (500 Hz) - after filtering to avoid aliasing
4. Segmentation to fixed length (10 seconds = 5000 samples at 500 Hz)
   - Zero-padding for shorter signals
   - Center crop for longer signals
5. Z-score normalization per lead
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal as scipy_signal


def resample_signal(
    x: np.ndarray,
    fs_original: float,
    fs_target: float = 500.0,
) -> np.ndarray:
    """Resample ECG signal to target sampling rate.
    
    Args:
        x: ECG signal array, shape (T, C) where T is time samples, C is channels/leads.
        fs_original: Original sampling rate in Hz.
        fs_target: Target sampling rate in Hz (default: 500.0).
    
    Returns:
        Resampled signal, shape (T_new, C) where T_new = int(T * fs_target / fs_original).
    """
    if fs_original == fs_target:
        return x
    
    # Calculate number of samples after resampling
    num_samples_original = x.shape[0]
    num_samples_target = int(num_samples_original * fs_target / fs_original)
    
    # Resample each lead separately
    if x.ndim == 1:
        # Single channel
        x_resampled = scipy_signal.resample(x, num_samples_target, axis=0)
    else:
        # Multi-channel: resample along time axis (axis=0)
        x_resampled = scipy_signal.resample(x, num_samples_target, axis=0)
    
    return x_resampled


def apply_bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth bandpass filter to ECG signal.
    
    Uses filtfilt for zero-phase filtering (no edge effects compensation needed).
    
    Args:
        x: ECG signal array, shape (T, C) where T is time samples, C is channels/leads.
        fs: Sampling rate in Hz.
        lowcut: Low cutoff frequency in Hz (default: 0.5).
        highcut: High cutoff frequency in Hz (default: 50.0).
        order: Filter order (default: 4).
    
    Returns:
        Filtered signal, same shape as input.
    """
    # Design Butterworth bandpass filter
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure frequencies are within valid range [0, 1]
    low = max(0.0, min(low, 1.0))
    high = max(0.0, min(high, 1.0))
    
    if low >= high:
        raise ValueError(f"Invalid frequency range: lowcut={lowcut} Hz, highcut={highcut} Hz (nyquist={nyquist} Hz)")
    
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    
    # Apply filtfilt (zero-phase filtering) to each lead
    if x.ndim == 1:
        # Single channel
        x_filtered = scipy_signal.filtfilt(b, a, x, axis=0)
    else:
        # Multi-channel: filter each lead separately
        x_filtered = np.zeros_like(x)
        for c in range(x.shape[1]):
            x_filtered[:, c] = scipy_signal.filtfilt(b, a, x[:, c], axis=0)
    
    return x_filtered


def apply_notch_filter(
    x: np.ndarray,
    fs: float,
    notch_freq: float = 60.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply notch filter to remove power line interference.
    
    For MIMIC data from USA, use 60 Hz (US power line frequency).
    Uses filtfilt for zero-phase filtering.
    
    Args:
        x: ECG signal array, shape (T, C) where T is time samples, C is channels/leads.
        fs: Sampling rate in Hz.
        notch_freq: Notch frequency in Hz (default: 60.0 for US power line).
        quality_factor: Quality factor for notch filter (default: 30.0).
                      Higher Q = narrower notch.
    
    Returns:
        Filtered signal, same shape as input.
    """
    # Check if notch frequency is within valid range
    nyquist = fs / 2.0
    if notch_freq >= nyquist:
        # Notch frequency too high, skip filtering
        return x
    
    # Design notch filter (bandstop filter at notch_freq)
    b, a = scipy_signal.iirnotch(notch_freq, quality_factor, fs)
    
    # Apply filtfilt (zero-phase filtering) to each lead
    if x.ndim == 1:
        # Single channel
        x_filtered = scipy_signal.filtfilt(b, a, x, axis=0)
    else:
        # Multi-channel: filter each lead separately
        x_filtered = np.zeros_like(x)
        for c in range(x.shape[1]):
            x_filtered[:, c] = scipy_signal.filtfilt(b, a, x[:, c], axis=0)
    
    return x_filtered


def segment_signal(
    x: np.ndarray,
    target_length: int = 5000,
    mode: str = "center_crop_pad",
) -> np.ndarray:
    """Segment ECG signal to fixed length.
    
    Args:
        x: ECG signal array, shape (T, C) where T is time samples, C is channels/leads.
        target_length: Target number of samples (default: 5000 for 10s at 500 Hz).
        mode: Segmentation mode:
            - "center_crop_pad": Center crop if longer, zero-pad if shorter (default).
    
    Returns:
        Segmented signal, shape (target_length, C).
    """
    current_length = x.shape[0]
    
    if current_length == target_length:
        return x
    
    if current_length < target_length:
        # Zero-padding: pad at the end
        if x.ndim == 1:
            padding = np.zeros(target_length - current_length, dtype=x.dtype)
            x_segmented = np.concatenate([x, padding])
        else:
            padding = np.zeros((target_length - current_length, x.shape[1]), dtype=x.dtype)
            x_segmented = np.vstack([x, padding])
    else:
        # Center crop: take middle portion
        start_idx = (current_length - target_length) // 2
        end_idx = start_idx + target_length
        x_segmented = x[start_idx:end_idx]
    
    return x_segmented


def normalize_zscore(
    x: np.ndarray,
    per_lead: bool = True,
) -> np.ndarray:
    """Apply Z-score normalization to ECG signal.
    
    Args:
        x: ECG signal array, shape (T, C) where T is time samples, C is channels/leads.
        per_lead: If True, normalize each lead separately (default: True).
                  If False, normalize across all leads.
    
    Returns:
        Normalized signal, same shape as input.
    """
    if per_lead:
        # Normalize each lead separately
        if x.ndim == 1:
            mean = np.mean(x)
            std = np.std(x)
            if std > 0:
                x_normalized = (x - mean) / std
            else:
                x_normalized = x - mean  # Avoid division by zero
        else:
            x_normalized = np.zeros_like(x)
            for c in range(x.shape[1]):
                mean = np.mean(x[:, c])
                std = np.std(x[:, c])
                if std > 0:
                    x_normalized[:, c] = (x[:, c] - mean) / std
                else:
                    x_normalized[:, c] = x[:, c] - mean  # Avoid division by zero
    else:
        # Normalize across all leads
        mean = np.mean(x)
        std = np.std(x)
        if std > 0:
            x_normalized = (x - mean) / std
        else:
            x_normalized = x - mean
    
    return x_normalized


def preprocess_ecg_signal(
    x: np.ndarray,
    fs: float,
    target_fs: float = 500.0,
    window_seconds: float = 10.0,
    filter_lowcut: float = 0.5,
    filter_highcut: float = 50.0,
    filter_order: int = 4,
    notch_freq: float = 60.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, float]:
    """Complete ECG preprocessing pipeline.
    
    Pipeline order (filtering before resampling to avoid aliasing):
    1. Bandpass filtering (filter_lowcut - filter_highcut Hz, Butterworth filter_order)
    2. Notch filtering (notch_freq Hz) to remove power line interference
    3. Resampling to target_fs (after filtering to avoid aliasing)
    4. Segmentation to fixed length (window_seconds * target_fs samples)
    5. Z-score normalization per lead (if normalize=True)
    
    Args:
        x: ECG signal array, shape (T, C) where T is time samples, C is channels/leads.
        fs: Original sampling rate in Hz.
        target_fs: Target sampling rate in Hz (default: 500.0).
        window_seconds: Target window length in seconds (default: 10.0).
        filter_lowcut: Low cutoff frequency in Hz (default: 0.5).
        filter_highcut: High cutoff frequency in Hz (default: 50.0).
        filter_order: Butterworth filter order (default: 4).
        notch_freq: Notch frequency in Hz for power line removal (default: 60.0 for US).
        normalize: Whether to apply Z-score normalization (default: True).
    
    Returns:
        Tuple of (preprocessed_signal, effective_fs).
        preprocessed_signal: shape (target_length, C) where target_length = int(window_seconds * target_fs).
        effective_fs: Effective sampling rate after resampling (should equal target_fs).
    """
    # Step 1: Bandpass filtering (on original sampling rate to avoid aliasing)
    x_processed = apply_bandpass_filter(
        x,
        fs=fs,
        lowcut=filter_lowcut,
        highcut=filter_highcut,
        order=filter_order,
    )
    
    # Step 2: Notch filtering (remove power line interference at 60 Hz for US data)
    x_processed = apply_notch_filter(
        x_processed,
        fs=fs,
        notch_freq=notch_freq,
    )
    
    # Step 3: Resampling (after filtering to avoid aliasing)
    x_processed = resample_signal(x_processed, fs, target_fs)
    effective_fs = target_fs
    
    # Step 4: Segmentation to fixed length
    target_length = int(window_seconds * target_fs)
    x_processed = segment_signal(x_processed, target_length=target_length)
    
    # Step 5: Z-score normalization per lead
    if normalize:
        x_processed = normalize_zscore(x_processed, per_lead=True)
    
    return x_processed, effective_fs


def create_preprocessing_pipeline(
    target_fs: float = 500.0,
    window_seconds: float = 10.0,
    filter_lowcut: float = 0.5,
    filter_highcut: float = 50.0,
    filter_order: int = 4,
    notch_freq: float = 60.0,
    normalize: bool = True,
):
    """Create a preprocessing function compatible with ECGDemoDataset.
    
    Returns a callable that can be used as the `preprocess` argument in ECGDemoDataset.
    The function signature is: (x, fs) -> x_processed
    
    Args:
        target_fs: Target sampling rate in Hz (default: 500.0).
        window_seconds: Target window length in seconds (default: 10.0).
        filter_lowcut: Low cutoff frequency in Hz (default: 0.5).
        filter_highcut: High cutoff frequency in Hz (default: 50.0).
        filter_order: Butterworth filter order (default: 4).
        notch_freq: Notch frequency in Hz for power line removal (default: 60.0 for US).
        normalize: Whether to apply Z-score normalization (default: True).
    
    Returns:
        Preprocessing function with signature (x, fs) -> x_processed.
    """
    def preprocess_fn(x: np.ndarray, fs: float) -> np.ndarray:
        """Preprocessing function compatible with ECGDemoDataset.
        
        Args:
            x: ECG signal array, shape (T, C).
            fs: Original sampling rate in Hz.
        
        Returns:
            Preprocessed signal, shape (target_length, C).
        """
        x_processed, _ = preprocess_ecg_signal(
            x=x,
            fs=fs,
            target_fs=target_fs,
            window_seconds=window_seconds,
            filter_lowcut=filter_lowcut,
            filter_highcut=filter_highcut,
            filter_order=filter_order,
            notch_freq=notch_freq,
            normalize=normalize,
        )
        return x_processed
    
    return preprocess_fn


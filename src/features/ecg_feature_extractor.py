"""Hand-crafted feature extraction for ECG signals.

Extracts statistical and frequency-domain features from ECG signals
for use with classical ML models (e.g., XGBoost).
"""

from typing import Optional
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq


def extract_handcrafted_features(
    ecg_signal: np.ndarray,
    fs: float = 500.0,
) -> np.ndarray:
    """
    Extract hand-crafted features from ECG signal.
    
    Extracts statistical and frequency-domain features per lead,
    plus cross-lead correlation features.
    
    Args:
        ecg_signal: ECG signal array, shape (12, 5000) or (5000, 12)
                   where 12 is number of leads, 5000 is time samples.
        fs: Sampling rate in Hz (default: 500.0).
    
    Returns:
        features: Feature vector of shape (n_features,)
                 ~300-400 features depending on number of leads.
    """
    # Ensure shape (T, C) = (5000, 12)
    if ecg_signal.shape[0] < ecg_signal.shape[1]:
        ecg_signal = ecg_signal.T  # (12, 5000) -> (5000, 12)
    
    num_leads = ecg_signal.shape[1]
    num_samples = ecg_signal.shape[0]
    
    all_features = []
    
    # Features per Lead
    for lead_idx in range(num_leads):
        lead_signal = ecg_signal[:, lead_idx]
        
        features_lead = []
        
        # === Time-Domain Features ===
        
        # Statistical Features
        features_lead.append(np.mean(lead_signal))           # Mean
        features_lead.append(np.std(lead_signal))             # Std
        features_lead.append(np.min(lead_signal))              # Min
        features_lead.append(np.max(lead_signal))              # Max
        features_lead.append(np.median(lead_signal))           # Median
        features_lead.append(stats.skew(lead_signal))         # Skewness
        features_lead.append(stats.kurtosis(lead_signal))     # Kurtosis
        features_lead.append(np.percentile(lead_signal, 25))   # Q1
        features_lead.append(np.percentile(lead_signal, 75))  # Q3
        features_lead.append(np.percentile(lead_signal, 90))  # P90
        features_lead.append(np.percentile(lead_signal, 10))  # P10
        
        # Range Features
        features_lead.append(np.max(lead_signal) - np.min(lead_signal))  # Range
        features_lead.append(np.percentile(lead_signal, 75) - np.percentile(lead_signal, 25))  # IQR
        
        # Energy Features
        features_lead.append(np.sum(lead_signal ** 2))         # Energy
        features_lead.append(np.mean(np.abs(lead_signal)))     # Mean Absolute Value
        features_lead.append(np.sqrt(np.mean(lead_signal ** 2)))  # RMS
        
        # Zero Crossing Rate
        zero_crossings = np.where(np.diff(np.signbit(lead_signal)))[0]
        features_lead.append(len(zero_crossings) / num_samples)  # ZCR
        
        # === Frequency-Domain Features ===
        # FFT
        fft_vals = np.abs(fft(lead_signal))
        fft_freqs = fftfreq(num_samples, 1/fs)
        
        # Positive frequencies only
        positive_freq_idx = fft_freqs > 0
        fft_vals = fft_vals[positive_freq_idx]
        fft_freqs = fft_freqs[positive_freq_idx]
        
        if len(fft_vals) > 0:
            # Dominant frequency
            dominant_freq_idx = np.argmax(fft_vals)
            features_lead.append(fft_freqs[dominant_freq_idx])     # Dominant frequency
            
            # Spectral power in different bands
            # Very Low Frequency (VLF): 0.04-0.15 Hz
            vlf_mask = (fft_freqs >= 0.04) & (fft_freqs < 0.15)
            features_lead.append(np.sum(fft_vals[vlf_mask] ** 2))  # VLF Power
            
            # Low Frequency (LF): 0.15-0.4 Hz
            lf_mask = (fft_freqs >= 0.15) & (fft_freqs < 0.4)
            features_lead.append(np.sum(fft_vals[lf_mask] ** 2))   # LF Power
            
            # High Frequency (HF): 0.4-2.0 Hz
            hf_mask = (fft_freqs >= 0.4) & (fft_freqs < 2.0)
            features_lead.append(np.sum(fft_vals[hf_mask] ** 2))   # HF Power
            
            # Total spectral power
            features_lead.append(np.sum(fft_vals ** 2))            # Total Power
            
            # Spectral centroid (weighted average frequency)
            if np.sum(fft_vals) > 0:
                spectral_centroid = np.sum(fft_freqs * fft_vals) / np.sum(fft_vals)
            else:
                spectral_centroid = 0.0
            features_lead.append(spectral_centroid)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum_energy = np.cumsum(fft_vals ** 2)
            total_energy = cumsum_energy[-1]
            if total_energy > 0:
                rolloff_idx = np.where(cumsum_energy >= 0.85 * total_energy)[0]
                if len(rolloff_idx) > 0:
                    features_lead.append(fft_freqs[rolloff_idx[0]])
                else:
                    features_lead.append(fft_freqs[-1])
            else:
                features_lead.append(0.0)
        else:
            # If no valid frequencies, add zeros
            features_lead.extend([0.0] * 8)
        
        all_features.extend(features_lead)
    
    # === Cross-Lead Features ===
    # Correlation between leads
    if num_leads > 1:
        # Correlation matrix (upper triangle)
        corr_matrix = np.corrcoef(ecg_signal.T)
        upper_triangle = corr_matrix[np.triu_indices(num_leads, k=1)]
        all_features.extend(upper_triangle)
    
    return np.array(all_features)


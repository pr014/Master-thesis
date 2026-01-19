"""ECG visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def plot_12lead_ecg(
    signal: np.ndarray,
    fs: float,
    lead_names: Optional[list[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (12, 16),
    show: bool = True,
) -> None:
    """Plot a 12-lead ECG signal.

    Args:
        signal: ECG signal array of shape (T, C) where T is time samples and C is channels/leads
        fs: Sampling frequency in Hz
        lead_names: List of lead names (default: ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
        title: Plot title
        output_path: Optional path to save the figure
        figsize: Figure size (width, height)
        show: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    if signal.ndim != 2:
        raise ValueError(f"Expected 2D array (T, C), got shape {signal.shape}")

    num_samples, num_leads = signal.shape

    if lead_names is None:
        default_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_names = default_leads[:num_leads] if num_leads <= len(default_leads) else [f"Lead {i+1}" for i in range(num_leads)]
    elif len(lead_names) != num_leads:
        raise ValueError(f"Number of lead names ({len(lead_names)}) must match number of channels ({num_leads})")

    # Time axis
    t = np.arange(num_samples) / fs

    # Create subplots
    fig, axes = plt.subplots(num_leads, 1, figsize=figsize, sharex=True)
    if num_leads == 1:
        axes = [axes]

    # Plot each lead
    for i in range(num_leads):
        axes[i].plot(t, signal[:, i], linewidth=0.8)
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, t[-1])

    axes[-1].set_xlabel("Time (s)")

    if title:
        fig.suptitle(title, y=0.98)
    else:
        duration = num_samples / fs
        fig.suptitle(f"12-lead ECG ({duration:.1f} s, {fs} Hz)", y=0.98)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_lead(
    signal: np.ndarray,
    fs: float,
    lead_idx: int = 0,
    lead_name: Optional[str] = None,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (10, 4),
    show: bool = True,
) -> None:
    """Plot a single ECG lead.

    Args:
        signal: ECG signal array of shape (T, C) or (T,)
        fs: Sampling frequency in Hz
        lead_idx: Index of the lead to plot (if signal is 2D)
        lead_name: Name of the lead
        title: Plot title
        output_path: Optional path to save the figure
        figsize: Figure size (width, height)
        show: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    # Handle 1D or 2D input
    if signal.ndim == 1:
        signal_1d = signal
    elif signal.ndim == 2:
        signal_1d = signal[:, lead_idx]
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {signal.shape}")

    num_samples = len(signal_1d)
    t = np.arange(num_samples) / fs

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(t, signal_1d, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(lead_name or f"Lead {lead_idx + 1}")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)
    else:
        duration = num_samples / fs
        ax.set_title(f"ECG Lead ({duration:.1f} s, {fs} Hz)")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


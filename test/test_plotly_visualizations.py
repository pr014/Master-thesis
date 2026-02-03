#!/usr/bin/env python3
"""
Test script for Plotly visualizations.
Demonstrates interactive plotting capabilities for ECG data analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_ecg_signal_plot():
    """Create an interactive ECG signal visualization."""
    print("Creating interactive ECG signal plot...")
    
    # Simulate ECG signal (12 leads)
    sampling_rate = 500  # Hz
    duration = 5  # seconds
    t = np.linspace(0, duration, sampling_rate * duration)
    
    # Generate synthetic ECG-like signals for different leads
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=leads,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    for idx, lead in enumerate(leads):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        # Generate synthetic ECG signal with QRS complexes
        signal = np.zeros_like(t)
        for i in range(5):  # 5 heartbeats
            heartbeat_start = i * 1.0  # 1 second per heartbeat (60 bpm)
            if heartbeat_start < duration:
                # QRS complex
                qrs_start = int(heartbeat_start * sampling_rate)
                qrs_end = int((heartbeat_start + 0.1) * sampling_rate)
                if qrs_end < len(signal):
                    signal[qrs_start:qrs_end] = np.sin(np.linspace(0, 2*np.pi, qrs_end - qrs_start)) * (1.5 + idx * 0.1)
                
                # P wave
                p_start = int((heartbeat_start - 0.15) * sampling_rate)
                p_end = int((heartbeat_start - 0.05) * sampling_rate)
                if p_start >= 0 and p_end < len(signal):
                    signal[p_start:p_end] = np.sin(np.linspace(0, np.pi, p_end - p_start)) * (0.3 + idx * 0.02)
        
        # Add noise
        noise = np.random.normal(0, 0.05, len(signal))
        signal += noise
        
        # Add baseline offset for different leads
        signal += idx * 0.5
        
        fig.add_trace(
            go.Scatter(
                x=t,
                y=signal,
                mode='lines',
                name=lead,
                line=dict(width=1.5, color=px.colors.qualitative.Set3[idx % 12]),
                hovertemplate=f'<b>{lead}</b><br>Time: %{{x:.3f}}s<br>Amplitude: %{{y:.2f}}mV<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        fig.update_yaxes(title_text="Amplitude (mV)", row=row, col=col)
    
    fig.update_layout(
        title={
            'text': 'Interactive 12-Lead ECG Signal Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_confusion_matrix_plot():
    """Create an interactive confusion matrix heatmap."""
    print("Creating interactive confusion matrix...")
    
    # Simulate confusion matrix (10 classes for LOS prediction)
    np.random.seed(42)
    cm = np.random.randint(0, 100, (10, 10))
    # Make it more diagonal (better predictions)
    for i in range(10):
        cm[i, i] = np.random.randint(80, 150)
    
    # Normalize to percentages
    cm_percent = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_percent,
        x=[f'Pred {i}' for i in range(10)],
        y=[f'True {i}' for i in range(10)],
        colorscale='Blues',
        text=cm_percent,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Percentage"),
        hovertemplate='<b>True: %{y}</b><br>Predicted: %{x}<br>Percentage: %{z:.1f}%<br>Count: %{customdata}<extra></extra>',
        customdata=cm
    ))
    
    fig.update_layout(
        title={
            'text': 'Interactive Confusion Matrix - LOS Prediction',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        width=800,
        height=700,
        template='plotly_white'
    )
    
    return fig


def create_training_curves_plot():
    """Create interactive training/validation curves."""
    print("Creating interactive training curves...")
    
    epochs = np.arange(1, 51)
    
    # Simulate training curves
    train_loss = 2.5 * np.exp(-epochs/15) + 0.3 + np.random.normal(0, 0.05, len(epochs))
    val_loss = 2.5 * np.exp(-epochs/12) + 0.4 + np.random.normal(0, 0.08, len(epochs))
    train_acc = 1 - (1 - 0.5) * np.exp(-epochs/20) + np.random.normal(0, 0.02, len(epochs))
    val_acc = 1 - (1 - 0.45) * np.exp(-epochs/18) + np.random.normal(0, 0.03, len(epochs))
    
    train_acc = np.clip(train_acc, 0, 1)
    val_acc = np.clip(val_acc, 0, 1)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Accuracy'),
        horizontal_spacing=0.15
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_loss,
            mode='lines+markers',
            name='Val Loss',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_acc,
            mode='lines+markers',
            name='Train Acc',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=4),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_acc,
            mode='lines+markers',
            name='Val Acc',
            line=dict(color='#d62728', width=2),
            marker=dict(size=4),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(
        title={
            'text': 'Interactive Training Curves',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=500,
        width=1200,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_class_distribution_plot():
    """Create interactive class distribution visualization."""
    print("Creating interactive class distribution...")
    
    # Simulate LOS class distribution
    classes = [f'Class {i}' for i in range(10)]
    counts = [1500, 1200, 800, 600, 400, 300, 200, 150, 100, 50]
    colors = px.colors.sequential.Blues_r
    
    fig = go.Figure()
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=classes,
        y=counts,
        marker=dict(
            color=counts,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Count")
        ),
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=[c/sum(counts)*100 for c in counts]
    ))
    
    fig.update_layout(
        title={
            'text': 'Interactive LOS Class Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='LOS Class',
        yaxis_title='Count',
        height=600,
        width=1000,
        template='plotly_white'
    )
    
    return fig


def main():
    """Generate all visualizations and save them."""
    output_dir = project_root / 'outputs' / 'visualizations' / 'plotly_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Plotly Visualization Test")
    print("=" * 60)
    print()
    
    # Create all visualizations
    visualizations = {
        'ecg_signal': create_ecg_signal_plot,
        'confusion_matrix': create_confusion_matrix_plot,
        'training_curves': create_training_curves_plot,
        'class_distribution': create_class_distribution_plot
    }
    
    for name, func in visualizations.items():
        try:
            fig = func()
            
            # Save as HTML (interactive)
            html_path = output_dir / f'{name}_interactive.html'
            fig.write_html(str(html_path))
            print(f"✓ Saved interactive HTML: {html_path}")
            
            # Save as static image (PNG)
            png_path = output_dir / f'{name}_static.png'
            fig.write_image(str(png_path), width=1200, height=800, scale=2)
            print(f"✓ Saved static PNG: {png_path}")
            print()
            
        except Exception as e:
            print(f"✗ Error creating {name}: {e}")
            print()
    
    print("=" * 60)
    print("All visualizations created successfully!")
    print(f"Output directory: {output_dir}")
    print()
    print("To view interactive plots, open the HTML files in a web browser.")
    print("=" * 60)


if __name__ == '__main__':
    main()


"""
Grad-CAM for 1D-CNN with visualization support.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def grad_cam_1d(model, x, class_idx):
    """
    x: tensor [1, n_leads, n_samples]
    class_idx: which output class to explain
    Returns: numpy array [n_samples] of importance values in [0, 1]
    """
    model.eval()
    x = x.clone().requires_grad_(False)
    
    logits, feat_map = model(x, return_features=True)
    score = logits[0, class_idx]
    
    grads = torch.autograd.grad(score, feat_map, retain_graph=False)[0]
    weights = grads.mean(dim=2, keepdim=True)
    
    cam = F.relu((weights * feat_map).sum(dim=1))
    cam = cam.squeeze(0).detach().numpy()
    
    n_samples = x.shape[-1]
    cam_upsampled = np.interp(
        np.linspace(0, len(cam) - 1, n_samples),
        np.arange(len(cam)),
        cam,
    )
    
    if cam_upsampled.max() > 0:
        cam_upsampled = cam_upsampled / cam_upsampled.max()
    return cam_upsampled


def top_attributed_region(cam, fs=100, top_fraction=0.15):
    """
    Returns (start_sec, end_sec, center_sec) for the most attributed region.
    """
    n = len(cam)
    k = max(1, int(n * top_fraction))
    window_scores = np.convolve(cam, np.ones(k) / k, mode="valid")
    center = np.argmax(window_scores) + k // 2
    start = max(0, center - k // 2)
    end = min(n, center + k // 2)
    return start / fs, end / fs, center / fs


def create_gradcam_visualization(signal, cam, fs, lead_idx=0, title="Grad-CAM Attribution"):
    """
    Creates a matplotlib figure showing the ECG waveform with Grad-CAM overlay.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Get the signal for the chosen lead
    if signal.ndim == 2:
        if lead_idx < signal.shape[1]:
            lead_signal = signal[:, lead_idx]
        else:
            lead_signal = signal[:, 0]
    else:
        lead_signal = signal
    
    # Time axis
    n_samples = len(lead_signal)
    time = np.arange(n_samples) / fs
    
    # Plot ECG waveform
    ax.plot(time, lead_signal, color='black', linewidth=1.5, label='ECG')
    
    # Plot Grad-CAM overlay
    # Normalize cam to match signal amplitude range
    signal_range = np.max(np.abs(lead_signal))
    if signal_range > 0:
        cam_scaled = cam * signal_range * 0.8
        ax.fill_between(time, -cam_scaled, cam_scaled, 
                        color='red', alpha=0.3, label='Grad-CAM Attribution')
    
    # Highlight the top region
    start_sec, end_sec, center_sec = top_attributed_region(cam, fs)
    ax.axvspan(start_sec, end_sec, color='red', alpha=0.2, label='Top Attributed Region')
    ax.axvline(center_sec, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_lead_comparison(signal, cam, fs, lead_names, title="Multi-Lead ECG with Grad-CAM"):
    """
    Creates a multi-lead visualization with Grad-CAM overlays.
    """
    n_leads = min(signal.shape[1], 6)  # Show up to 6 leads
    fig, axes = plt.subplots(n_leads, 1, figsize=(12, 3 * n_leads))
    
    if n_leads == 1:
        axes = [axes]
    
    time = np.arange(signal.shape[0]) / fs
    signal_range = np.max(np.abs(signal[:n_leads]))
    
    for i in range(n_leads):
        ax = axes[i]
        lead_name = lead_names[i] if lead_names else f"Lead {i+1}"
        
        ax.plot(time, signal[:, i], color='black', linewidth=1.0)
        
        if signal_range > 0:
            cam_scaled = cam * signal_range * 0.6
            ax.fill_between(time, -cam_scaled, cam_scaled, 
                            color='red', alpha=0.2)
        
        start_sec, end_sec, _ = top_attributed_region(cam, fs)
        ax.axvspan(start_sec, end_sec, color='red', alpha=0.1)
        
        ax.set_ylabel(lead_name)
        ax.set_xlim(0, time[-1])
        ax.grid(True, alpha=0.2)
        
        if i == 0:
            ax.set_title(title)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    return fig
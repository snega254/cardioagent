"""
Grad-CAM for the 1D-CNN. Produces a per-timestep importance curve over the
ECG waveform for a chosen predicted class, upsampled to input length.

This is the attribution method feeding both the final explanation and the
faithfulness sanity check in faithfulness.py.
"""
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam_1d(model, x, class_idx):
    """
    x: tensor [1, n_leads, n_samples]
    class_idx: which output class to explain
    Returns: numpy array [n_samples] of importance values in [0, 1]
    """
    model.eval()
    x = x.clone().requires_grad_(False)

    logits, feat_map = model(x, return_features=True)  # feat_map: [1, C, T']
    score = logits[0, class_idx]

    grads = torch.autograd.grad(score, feat_map, retain_graph=False)[0]  # [1, C, T']
    weights = grads.mean(dim=2, keepdim=True)  # global-average-pooled gradient, per channel

    cam = F.relu((weights * feat_map).sum(dim=1))  # [1, T']
    cam = cam.squeeze(0).detach().numpy()

    # Upsample from feature-map resolution to input sample resolution
    n_samples = x.shape[-1]
    cam_upsampled = np.interp(
        np.linspace(0, len(cam) - 1, n_samples),
        np.arange(len(cam)),
        cam,
    )

    if cam_upsampled.max() > 0:
        cam_upsampled = cam_upsampled / cam_upsampled.max()
    return cam_upsampled


def top_attributed_region(cam, fs=100, top_fraction=0.1):
    """
    Returns an approximate (start_sec, end_sec) window covering the most
    attributed contiguous region — used to phrase the explanation in
    human-readable terms ("around second X-Y").
    """
    n = len(cam)
    k = max(1, int(n * top_fraction))
    window_scores = np.convolve(cam, np.ones(k) / k, mode="valid")
    center = np.argmax(window_scores) + k // 2
    start = max(0, center - k // 2)
    end = min(n, center + k // 2)
    return start / fs, end / fs

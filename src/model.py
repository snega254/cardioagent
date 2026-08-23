"""
Small 1D-CNN for multi-label PTB-XL superclass classification.
Sized deliberately small (few conv blocks, global pooling) so it trains
in minutes on CPU with 8GB RAM. This is a design choice appropriate to
the time/hardware budget, not a claim of state-of-the-art performance —
say this explicitly in the paper's limitations section.
"""
import torch
import torch.nn as nn


class ECGConvNet(nn.Module):
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_leads, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Final conv layer — this is the layer Grad-CAM hooks into.
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, n_classes)

        # Single learned scalar for temperature scaling (post-hoc calibration).
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, return_features=False):
        feat_map = self.features(x)          # [B, 64, T']  <- Grad-CAM target
        pooled = self.gap(feat_map).squeeze(-1)  # [B, 64]
        logits = self.classifier(pooled)      # [B, n_classes]
        if return_features:
            return logits, feat_map
        return logits

    def calibrated_probs(self, logits):
        T = torch.exp(self.log_temperature)
        return torch.sigmoid(logits / T)

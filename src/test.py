"""
Test a trained ECGConvNet on the official PTB-XL test fold.

PTB-XL protocol:
    Folds 1-8  = training
    Fold 9     = validation
    Fold 10    = test

Usage from project root:

    python src/test.py

Or:

    python src/test.py --checkpoint checkpoint.pt
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PTBXLDataset, split_by_fold
from model import ECGConvNet
from preprocessing import load_metadata


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def evaluate(model, test_loader, device, temperature=1.0):

    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)

            # Get CNN output
            logits = model(x)

            # Apply temperature scaling
            logits = logits / temperature

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(y)

    # Combine batches
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()

    # Convert probabilities into 0/1 predictions
    predictions = (probs >= 0.5).astype(int)

    return predictions, probs, targets


def calculate_metrics(predictions, probs, targets):

    # Import metrics only here
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    print("\n" + "=" * 70)
    print("PTB-XL TEST RESULTS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"Number of test ECGs : {len(targets)}")
    print()
    print(f"Macro F1            : {macro_f1:.4f}")
    print(f"Micro F1            : {micro_f1:.4f}")
    print(f"Macro Precision     : {macro_precision:.4f}")
    print(f"Macro Recall        : {macro_recall:.4f}")

    # ---------------------------------------------------------
    # Overall AUROC
    # ---------------------------------------------------------

    try:

        macro_auc = roc_auc_score(
            targets,
            probs,
            average="macro"
        )

        print(f"Macro AUROC         : {macro_auc:.4f}")

    except ValueError:

        print("Macro AUROC         : N/A")

    # ---------------------------------------------------------
    # Per-class results
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("PER-CLASS RESULTS")
    print("-" * 70)

    for i, class_name in enumerate(SUPERCLASSES):

        f1 = f1_score(
            targets[:, i],
            predictions[:, i],
            zero_division=0
        )

        precision = precision_score(
            targets[:, i],
            predictions[:, i],
            zero_division=0
        )

        recall = recall_score(
            targets[:, i],
            predictions[:, i],
            zero_division=0
        )

        try:

            auc = roc_auc_score(
                targets[:, i],
                probs[:, i]
            )

            auc_text = f"{auc:.4f}"

        except ValueError:

            auc_text = "N/A"

        print(
            f"{class_name:5s} | "
            f"F1: {f1:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"AUROC: {auc_text}"
        )

    print("=" * 70)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data/ptbxl"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoint.pt"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device("cpu")

    print("Device:", device)

    # ---------------------------------------------------------
    # Load PTB-XL metadata
    # ---------------------------------------------------------

    print("\nLoading PTB-XL metadata...")

    df = load_metadata(args.data_dir)

    train_df, val_df, test_df = split_by_fold(df)

    print("Train records:", len(train_df))
    print("Validation records:", len(val_df))
    print("Test records:", len(test_df))

    # ---------------------------------------------------------
    # Create TEST dataset
    # ---------------------------------------------------------

    print("\nCreating test dataset...")

    test_ds = PTBXLDataset(
        test_df,
        args.data_dir
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    print("\nCreating ECGConvNet...")

    model = ECGConvNet().to(device)

    # ---------------------------------------------------------
    # Load trained checkpoint
    # ---------------------------------------------------------

    print("Loading checkpoint:", args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    # ---------------------------------------------------------
    # Load temperature
    # ---------------------------------------------------------

    temperature = 1.0

    if "log_temperature" in checkpoint:

        temperature = torch.exp(
            checkpoint["log_temperature"]
        ).item()

    print(f"Temperature: {temperature:.4f}")

    # ---------------------------------------------------------
    # Test model
    # ---------------------------------------------------------

    print("\nTesting on Fold 10...")

    predictions, probs, targets = evaluate(
        model,
        test_loader,
        device,
        temperature
    )

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    calculate_metrics(
        predictions,
        probs,
        targets
    )


if __name__ == "__main__":
    main()
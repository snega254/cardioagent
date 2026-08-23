"""
Evaluates the trained model on the official PTB-XL test fold (strat_fold==10).
Reports per-class and macro AUROC — this is the standard PTB-XL benchmark
metric, so your numbers are directly comparable to published baselines.

Run: python src/evaluate.py
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import PTBXLDataset, split_by_fold
from model import ECGConvNet
from preprocessing import SUPERCLASSES, load_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptbxl")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--max_records", type=int, default=None,
                         help="Leave unset to evaluate the full official test fold.")
    args = parser.parse_args()

    device = torch.device("cpu")
    df = load_metadata(args.data_dir)
    _, _, test_df = split_by_fold(df)
    test_ds = PTBXLDataset(test_df, args.data_dir, max_records=args.max_records)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = ECGConvNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.log_temperature.data = ckpt["log_temperature"]
    model.eval()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            probs = model.calibrated_probs(logits)
            all_probs.append(probs.numpy())
            all_targets.append(y.numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    print(f"Evaluated on {len(probs)} test records (official strat_fold==10).\n")
    aurocs = []
    for i, cls in enumerate(SUPERCLASSES):
        if targets[:, i].sum() == 0 or targets[:, i].sum() == len(targets):
            print(f"{cls}: AUROC undefined (only one class present in this subset)")
            continue
        auc = roc_auc_score(targets[:, i], probs[:, i])
        aurocs.append(auc)
        print(f"{cls}: AUROC = {auc:.4f}")

    if aurocs:
        print(f"\nMacro AUROC (over classes with both labels present): {np.mean(aurocs):.4f}")
    else:
        print("\nNo class had both labels present in this subset - "
              "increase --max_records or use the full test fold.")


if __name__ == "__main__":
    main()

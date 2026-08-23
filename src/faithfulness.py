"""
Lightweight faithfulness sanity check for Grad-CAM attributions.

Method (deletion test): zero out the top-attributed region of the input
and measure how much the predicted probability for the explained class
drops. Compare against zeroing out a random region of the same size.
If Grad-CAM is meaningfully faithful, the top-attributed deletion should
cause a LARGER probability drop than random deletion, on average.

This is a lightweight sanity check appropriate to a 7-hour budget — it is
NOT a full faithfulness validation (no perturbation-stability testing,
no comparison against multiple attribution methods). State this
explicitly as a limitation in the paper; do not oversell this result.

Run: python src/faithfulness.py
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PTBXLDataset, split_by_fold
from gradcam import grad_cam_1d
from model import ECGConvNet
from preprocessing import SUPERCLASSES, load_metadata


def deletion_drop(model, x, class_idx, cam, top_fraction=0.1, random_seed=0):
    n = x.shape[-1]
    k = int(n * top_fraction)

    with torch.no_grad():
        base_prob = model.calibrated_probs(model(x))[0, class_idx].item()

    # Top-attributed deletion
    top_idx = np.argsort(cam)[-k:]
    x_top_deleted = x.clone()
    x_top_deleted[0, :, top_idx] = 0.0
    with torch.no_grad():
        top_prob = model.calibrated_probs(model(x_top_deleted))[0, class_idx].item()

    # Random deletion (same amount removed, different location)
    rng = np.random.default_rng(random_seed)
    rand_idx = rng.choice(n, size=k, replace=False)
    x_rand_deleted = x.clone()
    x_rand_deleted[0, :, rand_idx] = 0.0
    with torch.no_grad():
        rand_prob = model.calibrated_probs(model(x_rand_deleted))[0, class_idx].item()

    return base_prob - top_prob, base_prob - rand_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptbxl")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--n_samples", type=int, default=100,
                         help="Number of test records to run the sanity check on.")
    args = parser.parse_args()

    device = torch.device("cpu")
    df = load_metadata(args.data_dir)
    _, _, test_df = split_by_fold(df)
    test_ds = PTBXLDataset(test_df, args.data_dir, max_records=args.n_samples)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = ECGConvNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.log_temperature.data = ckpt["log_temperature"]
    model.eval()

    top_drops, rand_drops = [], []
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
        pred_class = torch.argmax(logits[0]).item()

        cam = grad_cam_1d(model, x, pred_class)
        top_drop, rand_drop = deletion_drop(model, x, pred_class, cam)
        top_drops.append(top_drop)
        rand_drops.append(rand_drop)

    top_drops = np.array(top_drops)
    rand_drops = np.array(rand_drops)

    print(f"Ran deletion-test faithfulness sanity check on {len(top_drops)} test records.\n")
    print(f"Mean probability drop from deleting TOP-attributed region:    {top_drops.mean():.4f}")
    print(f"Mean probability drop from deleting RANDOM region (same size): {rand_drops.mean():.4f}")
    diff = top_drops.mean() - rand_drops.mean()
    print(f"\nDifference (top - random): {diff:.4f}")
    if diff > 0:
        print("Top-attributed deletion caused a larger average drop than random "
              "deletion - consistent with (but not proof of) faithful attribution.")
    else:
        print("Top-attributed deletion did NOT cause a larger drop than random "
              "deletion on this run - report this honestly; it indicates the "
              "attribution may not be reliably faithful on this model/data.")


if __name__ == "__main__":
    main()

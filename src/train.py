"""
Trains ECGConvNet on PTB-XL and fits post-hoc temperature scaling on the
validation fold. Defaults are chosen to realistically finish within a
small fraction of a 7-hour total project budget on CPU.

Run: python src/train.py
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PTBXLDataset, split_by_fold
from model import ECGConvNet
from preprocessing import load_metadata


def fit_temperature(model, val_loader, device):
    """Fit a single scalar temperature by minimizing NLL on val set logits."""
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / torch.exp(log_T), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_T.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptbxl")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_records", type=int, default=3000,
                         help="Cap training set size to guarantee finishing "
                              "within the time budget on CPU. Raise if you "
                              "have spare time — check printed epoch timing first.")
    parser.add_argument("--out", default="checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading metadata...")
    df = load_metadata(args.data_dir)
    train_df, val_df, test_df = split_by_fold(df)
    print(f"Train/Val/Test sizes (pre-cap): {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_ds = PTBXLDataset(train_df, args.data_dir, max_records=args.max_records)
    val_ds = PTBXLDataset(val_df, args.data_dir, max_records=min(500, len(val_df)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ECGConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{args.epochs} - loss {total_loss/len(train_ds):.4f} "
              f"- {elapsed:.1f}s "
              f"(~{elapsed*args.epochs/60:.1f} min for all epochs at this size)")

    print("Fitting temperature scaling on validation set...")
    log_T = fit_temperature(model, val_loader, device)
    model.log_temperature.data = log_T
    print(f"Fitted temperature: {torch.exp(log_T).item():.3f}")

    torch.save({"model_state": model.state_dict(),
                "log_temperature": log_T}, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()

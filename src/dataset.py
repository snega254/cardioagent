"""
PyTorch Dataset for PTB-XL, using the official strat_fold column for
patient-level splitting (folds 1-8 = train, 9 = val, 10 = test).
This is the standard PTB-XL benchmark protocol — using it means your
split is defensible without extra engineering, and avoids patient leakage.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import load_and_preprocess_record, multilabel_targets


class PTBXLDataset(Dataset):
    def __init__(self, df, data_dir, max_records=None):
        if max_records is not None:
            df = df.iloc[:max_records]
        self.df = df.reset_index()
        self.data_dir = data_dir
        self.y = multilabel_targets(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = load_and_preprocess_record(self.data_dir, row.filename_lr)
        y = self.y[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def split_by_fold(df):
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]
    return train_df, val_df, test_df

"""
Prints a handful of valid ecg_id values (from the official test fold) that
you can use with pipeline.py's --record_id argument.

Run: python src/list_valid_ids.py
"""
import argparse

from dataset import split_by_fold
from preprocessing import load_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptbxl")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    df = load_metadata(args.data_dir)
    _, _, test_df = split_by_fold(df)

    print(f"{args.n} valid ecg_id values from the test fold, with their "
          f"ground-truth superclasses (for your own reference only):\n")
    for ecg_id, row in test_df.head(args.n).iterrows():
        print(f"  {ecg_id}: {row.superclasses}")


if __name__ == "__main__":
    main()

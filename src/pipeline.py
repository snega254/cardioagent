"""
End-to-end demo: takes one PTB-XL record, runs it through the full
CardioAgent pipeline, and prints the final structured response.

Run: python src/pipeline.py --record_id <ecg_id>
If you don't know a valid id, run: python src/list_valid_ids.py
"""
import argparse

import torch

from gradcam import grad_cam_1d, top_attributed_region
from model import ECGConvNet
from preprocessing import SUPERCLASSES, load_and_preprocess_record, load_metadata
from rag import build_query, retrieve
from respond import CLASS_FULL_NAMES, compose_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptbxl")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--store_path", default="vector_store.pkl")
    parser.add_argument("--record_id", type=int, required=True)
    args = parser.parse_args()

    df = load_metadata(args.data_dir)
    if args.record_id not in df.index:
        sample_ids = list(df.index[:10])
        raise ValueError(
            f"record_id {args.record_id} not found. Either it doesn't exist in "
            f"the dataset, or it was filtered out for having no mapped "
            f"diagnostic superclass. Try one of these valid ids instead: "
            f"{sample_ids}\n(or run: python src/list_valid_ids.py)"
        )
    row = df.loc[args.record_id]

    x = load_and_preprocess_record(args.data_dir, row.filename_lr)
    x_tensor = torch.from_numpy(x).unsqueeze(0)

    model = ECGConvNet()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.log_temperature.data = ckpt["log_temperature"]
    model.eval()

    with torch.no_grad():
        logits = model(x_tensor)
        probs = model.calibrated_probs(logits)[0]

    pred_idx = torch.argmax(probs).item()
    pred_class = SUPERCLASSES[pred_idx]
    confidence = probs[pred_idx].item()

    cam = grad_cam_1d(model, x_tensor, pred_idx)
    start_sec, end_sec = top_attributed_region(cam, fs=100)

    query = build_query(pred_class, CLASS_FULL_NAMES[pred_class])
    passages = retrieve(query, store_path=args.store_path, top_k=3)

    response = compose_response(pred_class, confidence, start_sec, end_sec, passages)

    print("=" * 70)
    print(f"CardioAgent output for ecg_id={args.record_id}")
    print(f"(Ground-truth superclasses in dataset, for your own reference "
          f"only - do not present this as part of the model's output: "
          f"{row.superclasses})")
    print("=" * 70)
    print(response)


if __name__ == "__main__":
    main()

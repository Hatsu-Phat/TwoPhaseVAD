from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from phase1.models.paper_evit import PaperEViT


class FeatureDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root) / split
        self.split = split
        self.files = sorted(self.root.rglob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        x = torch.load(p, map_location="cpu")
        return p, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = PaperEViT(
        input_dim=int(cfg.get("input_dim", 1056)),
        seq_len=int(cfg.get("seq_len", 30)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 4)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.3)),
        num_classes=2,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    splits = ["train", "test"] if args.split == "all" else [args.split]

    for split in splits:
        ds = FeatureDataset(args.features_root, split)
        out_root = Path(args.out_dir) / split

        for feat_path, obj in tqdm(ds, desc=f"Infer {split}"):
            x = obj["features"].float().unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(x)

            video_score = float(out["video_scores"][0].cpu())
            segment_scores = out["segment_scores"][0].cpu()
            qap_weights = out["qap_weights"][0].cpu()

            rel_to_split = feat_path.relative_to(Path(args.features_root) / split)
            out_path = out_root / rel_to_split
            out_path.parent.mkdir(parents=True, exist_ok=True)

            save_obj = {
                "rel_path": obj.get("rel_path", ""),
                "class_name": obj.get("class_name", ""),
                "video_label": int(obj["video_label"]),
                "segment_scores": segment_scores,
                "qap_weights": qap_weights,
                "video_score": video_score,
                "selected_indices": obj["selected_indices"],
                "segment_bounds": obj.get("segment_bounds", torch.empty(0)),
                "fps": float(obj.get("fps", 30.0)),
                "total_frames": int(obj.get("total_frames", 0)),
                "feature_file": str(feat_path),
                "checkpoint": str(args.checkpoint),
            }

            torch.save(save_obj, out_path)

    print("DONE. saved to", args.out_dir)


if __name__ == "__main__":
    main()

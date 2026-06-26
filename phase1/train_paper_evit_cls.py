from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from phase1.models.paper_evit import PaperEViT


class FeatureDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root) / split
        self.files = sorted(self.root.rglob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        x = torch.load(p, map_location="cpu")
        feat = x["features"].float()
        label = int(x["video_label"])
        return {
            "features": feat,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(p),
            "rel_path": x.get("rel_path", ""),
            "class_name": x.get("class_name", ""),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "path": [b["path"] for b in batch],
        "rel_path": [b["rel_path"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ss = [], []

    for batch in loader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)

        out = model(x)
        score = out["video_scores"]

        ys.extend(y.cpu().numpy().tolist())
        ss.extend(score.cpu().numpy().tolist())

    auc = roc_auc_score(ys, ss) if len(set(ys)) > 1 else float("nan")
    ap = average_precision_score(ys, ss) if len(set(ys)) > 1 else float("nan")
    return auc, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FeatureDataset(cfg["features_root"], "train")
    test_ds = FeatureDataset(cfg["features_root"], "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=True,
    )

    model = PaperEViT(
        input_dim=int(cfg.get("input_dim", 1056)),
        seq_len=int(cfg.get("seq_len", 30)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 4)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.3)),
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    best_auc = -1.0
    history = []
    epochs = int(cfg.get("epochs", 80))

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            out = model(x)
            loss = criterion(out["video_logits"], y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.item()))

        video_auc, video_ap = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "video_auc": float(video_auc),
            "video_ap": float(video_ap),
        }
        history.append(row)

        print(json.dumps(row, ensure_ascii=False))

        if video_auc > best_auc:
            best_auc = video_auc
            ckpt = {
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "best_auc": best_auc,
                "model": "paper_evit",
            }
            torch.save(ckpt, out_dir / "best_auc.pth")
            print(f"Saved best checkpoint: video_auc={best_auc:.6f}")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print("DONE. best_video_auc:", best_auc)


if __name__ == "__main__":
    main()

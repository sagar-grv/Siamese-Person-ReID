"""Train and evaluate the paper-faithful Siamese ReID prototype.

Example:
    python train_siamese.py --data-root data/market1501_subset --epochs 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.reid_core import (
    IMAGE_SIZE,
    ContrastiveLoss,
    PairDataset,
    SiameseEncoder,
    embed_records,
    list_records,
    retrieval_metrics,
    write_gallery_json,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def export_onnx(model: nn.Module, path: Path, device: torch.device) -> None:
    model.eval()
    sample = torch.zeros(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)
    torch.onnx.export(
        model,
        sample,
        str(path),
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=18,
        external_data=False,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for left, right, label in loader:
        left, right, label = left.to(device), right.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(left), model(right), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/market1501_subset"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--pairs-per-epoch", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.artifacts.mkdir(parents=True, exist_ok=True)

    train_records = list_records(args.data_root / "train")
    query_records = list_records(args.data_root / "query")
    gallery_records = list_records(args.data_root / "gallery")
    train_pids = {record.pid for record in train_records}
    eval_pids = {record.pid for record in query_records} | {record.pid for record in gallery_records}
    assert train_pids.isdisjoint(eval_pids), "Evaluation identities must be disjoint from training identities"

    dataset = PairDataset(train_records, pairs_per_epoch=args.pairs_per_epoch, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SiameseEncoder(embedding_dim=args.embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_loss = float("inf")
    checkpoint_path = args.artifacts / "siamese_market1501.pt"
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": loss, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch:02d}/{args.epochs} loss={loss:.5f} lr={optimizer.param_groups[0]['lr']:.6f}")
        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "image_size": IMAGE_SIZE,
                    "margin": args.margin,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    query_embeddings = embed_records(model, query_records, device)
    gallery_embeddings = embed_records(model, gallery_records, device)
    metrics = retrieval_metrics(query_records, gallery_records, query_embeddings, gallery_embeddings)
    metrics.update(
        {
            "dataset": "Market-1501 subset",
            "train_images": len(train_records),
            "train_identities": len(train_pids),
            "eval_identities": len(eval_pids),
            "embedding_dim": args.embedding_dim,
            "seed": args.seed,
        }
    )
    (args.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.artifacts / "history.json").write_text(json.dumps(history, indent=2))
    write_gallery_json(args.artifacts / "gallery.json", gallery_records, gallery_embeddings)
    export_onnx(model.cpu(), args.artifacts / "siamese_encoder.onnx", torch.device("cpu"))
    print(json.dumps(metrics, indent=2))
    print(f"saved={checkpoint_path}")


if __name__ == "__main__":
    main()

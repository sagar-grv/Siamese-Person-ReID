"""Evaluate an existing Siamese checkpoint and export browser artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.reid_core import (
    IMAGE_SIZE,
    SiameseEncoder,
    embed_records,
    list_records,
    retrieval_metrics,
    write_gallery_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/market1501_subset"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    device = torch.device("cpu")
    checkpoint = torch.load(args.artifacts / "siamese_market1501.pt", map_location=device, weights_only=True)
    model = SiameseEncoder(embedding_dim=int(checkpoint["embedding_dim"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    query_records = list_records(args.data_root / "query")
    gallery_records = list_records(args.data_root / "gallery")
    query_embeddings = embed_records(model, query_records, device)
    gallery_embeddings = embed_records(model, gallery_records, device)
    metrics = retrieval_metrics(query_records, gallery_records, query_embeddings, gallery_embeddings)
    metrics.update({"dataset": "Market-1501 subset", "embedding_dim": int(checkpoint["embedding_dim"]), "seed": int(checkpoint["seed"])})
    (args.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_gallery_json(args.artifacts / "gallery.json", gallery_records, gallery_embeddings)
    model.eval()
    sample = torch.zeros(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    torch.onnx.export(
        model,
        sample,
        str(args.artifacts / "siamese_encoder.onnx"),
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=18,
        external_data=False,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

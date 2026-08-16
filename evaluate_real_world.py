"""Evaluate a manifest-backed ReID deployment and calibrate review thresholds.

Example:
    python3 evaluate_real_world.py \
      --manifest examples/site_manifest.csv \
      --model artifacts/upgraded/strong_reid_market1501.pt \
      --metrics-out reports/real_world_metrics.json \
      --version-out reports/gallery_version.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.realworld import gallery_version, load_manifest
from src.reid_core import ImageRecord
from train_upgraded import StrongReIDModel, embed_records, market_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--version-out", type=Path, required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--dataset-name", default="authorized-site-manifest")
    parser.add_argument("--model-name", default="strong-reid-resnet18")
    parser.add_argument("--negative-sample-limit", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def calibrate_thresholds(
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    negative_sample_limit: int,
    seed: int,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    positive: list[float] = []
    negative: list[float] = []
    for query, query_embedding in zip(query_records, query_embeddings):
        valid = np.array(
            [not (gallery.pid == query.pid and gallery.camid == query.camid) for gallery in gallery_records]
        )
        scores = gallery_embeddings[valid] @ query_embedding
        labels = np.array([gallery.pid == query.pid for gallery, keep in zip(gallery_records, valid) if keep])
        positive.extend(scores[labels].astype(float).tolist())
        negatives = scores[~labels].astype(float).tolist()
        if len(negative) < negative_sample_limit:
            remaining = negative_sample_limit - len(negative)
            negative.extend(negatives[:remaining])
        elif negatives:
            for value in negatives:
                index = rng.randrange(negative_sample_limit)
                negative[index] = float(value)
    if not positive or not negative:
        return {"status": "insufficient-labeled-pairs", "positive_pairs": len(positive), "negative_pairs": len(negative)}

    positive_array = np.asarray(positive)
    negative_array = np.asarray(negative)
    thresholds = np.unique(np.quantile(np.concatenate([positive_array, negative_array]), np.linspace(0.01, 0.99, 199)))
    selected = float(thresholds[0])
    selected_fpr = 1.0
    selected_tpr = 0.0
    for threshold in thresholds:
        fpr = float(np.mean(negative_array >= threshold))
        tpr = float(np.mean(positive_array >= threshold))
        if fpr <= 0.01 and tpr > selected_tpr:
            selected = float(threshold)
            selected_fpr = fpr
            selected_tpr = tpr
    return {
        "status": "calibrated",
        "target_false_positive_rate": 0.01,
        "review_threshold": selected,
        "validation_false_positive_rate": selected_fpr,
        "validation_true_positive_rate": selected_tpr,
        "positive_pairs": int(len(positive_array)),
        "negative_pairs": int(len(negative_array)),
        "positive_score_p50": float(np.quantile(positive_array, 0.50)),
        "negative_score_p95": float(np.quantile(negative_array, 0.95)),
    }


def main() -> None:
    args = parse_args()
    grouped = load_manifest(args.manifest, args.manifest_root)
    query_records = grouped.get("query", grouped.get("validation", []))
    gallery_records = grouped.get("gallery", [])
    if not query_records or not gallery_records:
        raise ValueError("Manifest must contain non-empty query and gallery or validation splits")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = StrongReIDModel(
        num_classes=int(checkpoint["train_identities"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    query_embeddings = embed_records(model, query_records, torch.device("cpu"), batch_size=128)
    gallery_embeddings = embed_records(model, gallery_records, torch.device("cpu"), batch_size=128)

    metrics: dict[str, object] = market_metrics(query_records, gallery_records, query_embeddings, gallery_embeddings)
    by_camera: dict[str, dict[str, float]] = {}
    camera_indices: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(query_records):
        camera_indices[record.camid].append(index)
    for camera_id, indices in sorted(camera_indices.items()):
        group_records = [query_records[index] for index in indices]
        group_embeddings = query_embeddings[indices]
        by_camera[str(camera_id)] = market_metrics(
            group_records, gallery_records, group_embeddings, gallery_embeddings
        )
    metrics["by_query_camera"] = by_camera
    metrics["threshold_calibration"] = calibrate_thresholds(
        query_records,
        gallery_records,
        query_embeddings,
        gallery_embeddings,
        args.negative_sample_limit,
        args.seed,
    )
    metrics["model_name"] = args.model_name
    metrics["dataset"] = args.dataset_name
    metrics["embedding_dim"] = int(checkpoint["embedding_dim"])
    metrics["query_split"] = "query" if grouped.get("query") else "validation"
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2))

    version = gallery_version(args.model, args.gallery_json, args.metrics_out, args.model_name, args.dataset_name)
    args.version_out.parent.mkdir(parents=True, exist_ok=True)
    args.version_out.write_text(json.dumps(version, indent=2))
    print(json.dumps({"metrics": metrics, "gallery_version": version}, indent=2))


if __name__ == "__main__":
    main()

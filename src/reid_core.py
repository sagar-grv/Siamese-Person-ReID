"""Core Siamese person re-identification components.

This module follows the classic Siamese formulation: both branches share the
same encoder, and a contrastive objective pulls same-identity pairs together
while pushing different-identity pairs apart.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset

IMAGE_SIZE = (128, 64)  # height, width; Market-1501 native aspect ratio
FILENAME_RE = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+).+\.jpg$")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    pid: int
    camid: int
    name: str


def parse_record(path: Path) -> ImageRecord:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected Market-1501 filename: {path.name}")
    return ImageRecord(
        path=path,
        pid=int(match.group("pid")),
        camid=int(match.group("cam")),
        name=path.name,
    )


def list_records(directory: str | Path) -> list[ImageRecord]:
    records = [parse_record(path) for path in sorted(Path(directory).glob("*.jpg"))]
    if not records:
        raise FileNotFoundError(f"No JPG images found in {directory}")
    return records


def image_to_tensor(path: Path, augment: bool = False) -> Tensor:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    if augment and random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32) / 255.0
    # ImageNet-style normalization keeps the input scale stable for training.
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).contiguous()


class SiameseEncoder(nn.Module):
    """Compact CNN encoder with shared weights on both Siamese branches."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        embeddings = self.projection(self.features(images))
        return nn.functional.normalize(embeddings, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss with label 0 for same identity and 1 for different."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, left: Tensor, right: Tensor, label: Tensor) -> Tensor:
        distance = nn.functional.pairwise_distance(left, right)
        same_loss = (1.0 - label) * distance.pow(2)
        different_loss = label * torch.relu(self.margin - distance).pow(2)
        return (same_loss + different_loss).mean()


class PairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """On-the-fly positive/negative pair generator for a directory of crops."""

    def __init__(self, records: Iterable[ImageRecord], pairs_per_epoch: int, seed: int = 42) -> None:
        self.records = list(records)
        self.by_pid: dict[int, list[ImageRecord]] = {}
        for record in self.records:
            self.by_pid.setdefault(record.pid, []).append(record)
        self.pids = sorted(self.by_pid)
        self.pairs_per_epoch = pairs_per_epoch
        self.seed = seed
        if len(self.pids) < 2:
            raise ValueError("PairDataset needs at least two identities")
        if any(len(items) < 2 for items in self.by_pid.values()):
            raise ValueError("Each training identity needs at least two images")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rng = random.Random(self.seed + index)
        same = index % 2 == 0
        if same:
            pid = rng.choice(self.pids)
            left, right = rng.sample(self.by_pid[pid], 2)
            label = 0.0
        else:
            pid_left, pid_right = rng.sample(self.pids, 2)
            left = rng.choice(self.by_pid[pid_left])
            right = rng.choice(self.by_pid[pid_right])
            label = 1.0
        return (
            image_to_tensor(left.path, augment=True),
            image_to_tensor(right.path, augment=True),
            torch.tensor(label, dtype=torch.float32),
        )


def embed_records(
    model: nn.Module,
    records: list[ImageRecord],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = torch.stack(
                [image_to_tensor(record.path) for record in records[start : start + batch_size]]
            ).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def retrieval_metrics(
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    top_k: int = 5,
) -> dict[str, float]:
    distances = 1.0 - query_embeddings @ gallery_embeddings.T
    top1: list[float] = []
    topk: list[float] = []
    average_precisions: list[float] = []
    for query, row in zip(query_records, distances):
        ranking = np.argsort(row)
        relevant = np.array([gallery_records[i].pid == query.pid for i in ranking])
        top1.append(float(relevant[:1].any()))
        topk.append(float(relevant[:top_k].any()))
        relevant_count = int(relevant.sum())
        if relevant_count == 0:
            average_precisions.append(0.0)
            continue
        precision = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        average_precisions.append(float((precision * relevant).sum() / relevant_count))
    return {
        "queries": float(len(query_records)),
        "gallery_images": float(len(gallery_records)),
        "top1_accuracy": float(np.mean(top1)),
        "top5_accuracy": float(np.mean(topk)),
        "mean_average_precision": float(np.mean(average_precisions)),
    }


def write_gallery_json(
    path: str | Path,
    records: list[ImageRecord],
    embeddings: np.ndarray,
    public_prefix: str = "/gallery",
) -> None:
    payload = [
        {
            "name": record.name,
            "pid": record.pid,
            "camid": record.camid,
            "image": f"{public_prefix}/{record.name}",
            "embedding": embedding.astype(float).round(7).tolist(),
        }
        for record, embedding in zip(records, embeddings)
    ]
    Path(path).write_text(json.dumps(payload, separators=(",", ":")))

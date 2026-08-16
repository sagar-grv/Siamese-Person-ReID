"""Real-world data and deployment utilities for person ReID.

The module keeps site-specific data in a manifest rather than encoding operational
metadata in filenames. It supports identity-balanced, cross-camera batches and
versioned gallery metadata without storing raw personal data in Git.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from torch.utils.data import Sampler

from src.reid_core import ImageRecord

REQUIRED_COLUMNS = {
    "image_path",
    "person_id",
    "camera_id",
    "session_id",
    "timestamp",
    "bbox_quality",
    "consent_status",
    "split",
}
VALID_SPLITS = {"train", "query", "gallery", "validation", "test"}


def load_manifest(path: str | Path, root: str | Path | None = None) -> dict[str, list[ImageRecord]]:
    """Load a CSV manifest and return records grouped by split.

    Paths are resolved relative to ``root`` when provided, otherwise relative to
    the manifest directory. The function rejects missing files, invalid splits,
    missing authorization status, and duplicate image paths.
    """
    manifest_path = Path(path).resolve()
    base = Path(root).resolve() if root else manifest_path.parent
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        grouped: dict[str, list[ImageRecord]] = defaultdict(list)
        seen_paths: set[Path] = set()
        for row_number, row in enumerate(reader, start=2):
            split = (row.get("split") or "").strip().lower()
            if split not in VALID_SPLITS:
                raise ValueError(f"Invalid split {split!r} at manifest row {row_number}")
            image_path = (base / row["image_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image at manifest row {row_number}: {image_path}")
            if image_path in seen_paths:
                raise ValueError(f"Duplicate image path at manifest row {row_number}: {image_path}")
            seen_paths.add(image_path)
            if not row["consent_status"].strip():
                raise ValueError(f"Missing consent_status at manifest row {row_number}")
            grouped[split].append(
                ImageRecord(
                    path=image_path,
                    pid=int(row["person_id"]),
                    camid=int(row["camera_id"]),
                    name=image_path.name,
                )
            )
    if not grouped.get("train"):
        raise ValueError("Manifest must contain a non-empty train split")
    return dict(grouped)


def write_market_manifest(
    output: str | Path,
    split_roots: dict[str, str | Path],
    authorization: str = "public-benchmark-license-check-required",
) -> None:
    """Create a manifest from Market-style directories for reproducible testing."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for split, directory in split_roots.items():
        for image_path in sorted(Path(directory).glob("*.jpg")):
            name = image_path.name
            parts = name.split("_c", 1)
            if len(parts) != 2:
                continue
            pid = parts[0]
            camid = parts[1].split("s", 1)[0]
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "person_id": pid,
                    "camera_id": camid,
                    "session_id": "benchmark",
                    "timestamp": "",
                    "bbox_quality": "unknown",
                    "consent_status": authorization,
                    "split": split,
                }
            )
    fieldnames = sorted(REQUIRED_COLUMNS | {"timestamp"})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CameraIdentityBatchSampler(Sampler[list[int]]):
    """Sample P identities x K images, preferring cross-camera positives."""

    def __init__(
        self,
        records: list[ImageRecord],
        identities_per_batch: int,
        images_per_identity: int,
        batches_per_epoch: int,
        seed: int = 42,
    ) -> None:
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        by_pid: dict[int, list[int]] = defaultdict(list)
        by_pid_camera: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(records):
            by_pid[record.pid].append(index)
            by_pid_camera[record.pid][record.camid].append(index)
        self.by_pid = dict(by_pid)
        self.by_pid_camera = {pid: dict(cameras) for pid, cameras in by_pid_camera.items()}
        self.pids = sorted(self.by_pid)
        if len(self.pids) < identities_per_batch:
            raise ValueError("Not enough identities for the requested balanced batch")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _sample_identity(self, pid: int, rng: random.Random) -> list[int]:
        cameras = self.by_pid_camera[pid]
        camera_ids = list(cameras)
        if len(camera_ids) >= 2 and self.images_per_identity >= 2:
            left_camera, right_camera = rng.sample(camera_ids, 2)
            batch = [rng.choice(cameras[left_camera]), rng.choice(cameras[right_camera])]
            remaining = self.images_per_identity - len(batch)
            choices = self.by_pid[pid]
            batch.extend(rng.choices(choices, k=remaining))
            return batch
        choices = self.by_pid[pid]
        if len(choices) >= self.images_per_identity:
            return rng.sample(choices, self.images_per_identity)
        return rng.choices(choices, k=self.images_per_identity)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        for _ in range(self.batches_per_epoch):
            pids = rng.sample(self.pids, self.identities_per_batch)
            batch: list[int] = []
            for pid in pids:
                batch.extend(self._sample_identity(pid, rng))
            yield batch


def gallery_version(
    model_path: str | Path,
    gallery_path: str | Path,
    metrics_path: str | Path,
    model_name: str,
    dataset_name: str,
) -> dict[str, object]:
    """Create a content-addressed metadata record for a deployed gallery."""
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    model_file = Path(model_path)
    gallery_file = Path(gallery_path)
    metrics_file = Path(metrics_path)
    gallery = json.loads(gallery_file.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    return {
        "version": f"{sha256(model_file)[:12]}-{sha256(gallery_file)[:12]}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset": dataset_name,
        "model_sha256": sha256(model_file),
        "gallery_sha256": sha256(gallery_file),
        "metrics_sha256": sha256(metrics_file),
        "gallery_images": len(gallery),
        "embedding_dim": len(gallery[0]["embedding"]) if gallery else 0,
        "evaluation": metrics,
    }

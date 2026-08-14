"""Utilities shared by the training entry points.

The repository intentionally does not commit the image dataset or generated
model artifacts. These helpers validate that the external files are present
before a training run starts, so failures are reported at the real cause.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ("Anchor", "Positive", "Negative")


def load_validated_triplets(csv_file: str | Path, data_dir: str | Path) -> pd.DataFrame:
    """Load and validate the triplet annotation CSV and referenced images.

    Raises:
        FileNotFoundError: If the CSV, image directory, or referenced images
            are missing.
        ValueError: If the CSV is empty or has an invalid schema.
    """

    csv_path = Path(csv_file)
    image_dir = Path(data_dir)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Triplet annotation file not found: {csv_path}. "
            """Download the dataset and place its train.csv at this path."""
        )

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}. "
            """Download/extract the dataset so the images are in data/SNN-TL-Data/train/."""
        )

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Invalid annotation schema in {csv_path}: missing columns "
            f"{', '.join(missing_columns)}. Expected: {', '.join(REQUIRED_COLUMNS)}."
        )

    if df.empty:
        raise ValueError(f"Annotation file is empty: {csv_path}")

    referenced_names = {
        str(name).strip()
        for column in REQUIRED_COLUMNS
        for name in df[column].dropna()
        if str(name).strip()
    }
    missing_images = sorted(
        name for name in referenced_names if not (image_dir / name).is_file()
    )
    if missing_images:
        preview = ", ".join(missing_images[:5])
        suffix = " ..." if len(missing_images) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing_images)} image(s) referenced by {csv_path} are missing "
            f"from {image_dir}. Examples: {preview}{suffix}"
        )

    return df


def ensure_directories(paths: Iterable[str | Path]) -> None:
    """Create directories used for checkpoints, databases, and plots."""

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

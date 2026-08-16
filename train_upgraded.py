"""Train a stronger person-ReID model on the full Market-1501 protocol.

The upgrade keeps the browser-facing normalized embedding interface but adds:
- a pretrained ResNet-18 backbone;
- identity-balanced P x K batches;
- batch-hard triplet loss;
- identity classification with label smoothing;
- person-ReID augmentations and warmup/cosine scheduling;
- standard same-camera gallery filtering for Market-1501 metrics.

Example:
    python train_upgraded.py \
      --market-root /home/ubuntu/datasets/Market-1501-v15.09.15 \
      --artifacts artifacts/upgraded \
      --epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.realworld import CameraIdentityBatchSampler, load_manifest
from src.reid_core import IMAGE_SIZE, ImageRecord, list_records, write_gallery_json


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ReIDImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, records: list[ImageRecord], augment: bool) -> None:
        self.records = records
        self.transform = self._build_transform(augment)

    @staticmethod
    def _build_transform(augment: bool) -> transforms.Compose:
        if augment:
            return transforms.Compose(
                [
                    transforms.Resize((144, 72)),
                    transforms.RandomCrop(IMAGE_SIZE),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                    transforms.RandomErasing(p=0.35, scale=(0.02, 0.18), ratio=(0.3, 3.3)),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.pid


class StrongReIDModel(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, images: Tensor, return_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        features = self.backbone(images)
        metric_features = self.bnneck(features)
        embeddings = nn.functional.normalize(self.projection(metric_features), p=2, dim=1)
        if return_logits:
            return embeddings, self.classifier(metric_features)
        return embeddings


def batch_hard_triplet_loss(embeddings: Tensor, labels: Tensor, margin: float = 0.3) -> Tensor:
    distances = 1.0 - embeddings @ embeddings.t()
    same = labels[:, None].eq(labels[None, :])
    positive_mask = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative_mask = ~same
    hardest_positive = distances.masked_fill(~positive_mask, -1e4).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, 1e4).min(dim=1).values
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    return nn.functional.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def market_metrics(
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    top_k: int = 5,
) -> dict[str, float]:
    top1: list[float] = []
    topk: list[float] = []
    average_precisions: list[float] = []
    for query, query_embedding in zip(query_records, query_embeddings):
        valid = np.array(
            [not (gallery.pid == query.pid and gallery.camid == query.camid) for gallery in gallery_records]
        )
        candidates = np.flatnonzero(valid)
        scores = gallery_embeddings[candidates] @ query_embedding
        ranking = candidates[np.argsort(-scores)]
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


def embed_records(model: nn.Module, records: list[ImageRecord], device: torch.device, batch_size: int) -> np.ndarray:
    dataset = ReIDImageDataset(records, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            outputs.append(model(images.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def export_onnx(model: StrongReIDModel, path: Path, device: torch.device) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--market-root", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--demo-root", type=Path, default=Path("data/market1501_subset"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/upgraded"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches-per-epoch", type=int, default=200)
    parser.add_argument("--identities-per-batch", type=int, default=8)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-hard-margin", type=float, default=0.3)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.artifacts.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        grouped = load_manifest(args.manifest, args.manifest_root)
        train_records = grouped["train"]
        query_records = grouped.get("query", grouped.get("validation", []))
        gallery_records = grouped.get("gallery", grouped.get("test", []))
        dataset_name = f"manifest:{args.manifest.name}"
        if not query_records or not gallery_records:
            raise ValueError("A manifest training run requires non-empty query and gallery or validation splits")
    else:
        train_records = list_records(args.market_root / "bounding_box_train")
        query_records = list_records(args.market_root / "query")
        gallery_records = list_records(args.market_root / "bounding_box_test")
        dataset_name = "Market-1501 full official split"
    train_pids = {record.pid for record in train_records}
    eval_pids = {record.pid for record in query_records} | {record.pid for record in gallery_records}
    identity_overlap = sorted(train_pids & eval_pids)
    if not args.manifest and identity_overlap:
        raise ValueError(f"Market-1501 train/eval identity overlap detected: {identity_overlap[:10]}")

    # Market-1501 uses different pid ranges for train and test, so class indices are local to training.
    pid_to_class = {pid: index for index, pid in enumerate(sorted(train_pids))}
    sampler = CameraIdentityBatchSampler(
        train_records,
        identities_per_batch=args.identities_per_batch,
        images_per_identity=args.images_per_identity,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )
    train_dataset = ReIDImageDataset(train_records, augment=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=2, pin_memory=device.type == "cuda")
    model = StrongReIDModel(
        num_classes=len(pid_to_class),
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.learning_rate * 0.02
    )

    best_loss = float("inf")
    history: list[dict[str, float]] = []
    checkpoint_path = args.artifacts / "strong_reid_market1501.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch <= args.warmup_epochs:
            scale = epoch / max(1, args.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
        total_loss = 0.0
        total_triplet = 0.0
        total_classification = 0.0
        for images, raw_labels in train_loader:
            labels = torch.tensor([pid_to_class[int(pid)] for pid in raw_labels], dtype=torch.long, device=device)
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(images, return_logits=True)
            triplet = batch_hard_triplet_loss(embeddings, labels, margin=args.batch_hard_margin)
            classification = criterion(logits, labels)
            loss = args.triplet_weight * triplet + args.classification_weight * classification
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_triplet += float(triplet.item())
            total_classification += float(classification.item())
        if epoch > args.warmup_epochs:
            scheduler.step()
        record = {
            "epoch": float(epoch),
            "loss": total_loss / len(train_loader),
            "triplet_loss": total_triplet / len(train_loader),
            "classification_loss": total_classification / len(train_loader),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"epoch={epoch:02d}/{args.epochs} loss={record['loss']:.4f} "
            f"triplet={record['triplet_loss']:.4f} cls={record['classification_loss']:.4f} "
            f"lr={record['learning_rate']:.7f}",
            flush=True,
        )
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "image_size": IMAGE_SIZE,
                    "seed": args.seed,
                    "train_identities": len(train_pids),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    query_embeddings = embed_records(model, query_records, device, batch_size=128)
    gallery_embeddings = embed_records(model, gallery_records, device, batch_size=128)
    metrics = market_metrics(query_records, gallery_records, query_embeddings, gallery_embeddings)
    metrics.update(
        {
            "dataset": dataset_name,
            "train_images": len(train_records),
            "train_identities": len(train_pids),
            "eval_identities": len(eval_pids),
            "train_eval_identity_overlap": len(identity_overlap),
            "embedding_dim": args.embedding_dim,
            "backbone": "pretrained-resnet18",
            "loss": "cross-entropy + batch-hard triplet",
            "augmentations": "random-crop, horizontal-flip, color-jitter, random-erasing",
            "epochs": args.epochs,
            "batches_per_epoch": args.batches_per_epoch,
            "seed": args.seed,
        }
    )
    (args.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.artifacts / "history.json").write_text(json.dumps(history, indent=2))

    # Generate browser assets from the demo subset when available; otherwise use manifest query/gallery records.
    if (args.demo_root / "query").is_dir() and (args.demo_root / "gallery").is_dir():
        demo_query = list_records(args.demo_root / "query")
        demo_gallery = list_records(args.demo_root / "gallery")
    else:
        demo_query = query_records
        demo_gallery = gallery_records
    demo_query_embeddings = embed_records(model, demo_query, device, batch_size=128)
    demo_gallery_embeddings = embed_records(model, demo_gallery, device, batch_size=128)
    write_gallery_json(args.artifacts / "gallery.json", demo_gallery, demo_gallery_embeddings)
    export_onnx(model.cpu(), args.artifacts / "siamese_encoder.onnx", torch.device("cpu"))
    print(json.dumps(metrics, indent=2))
    print(f"saved={checkpoint_path}")
    print(f"demo_queries={len(demo_query)} demo_gallery={len(demo_gallery)}")


if __name__ == "__main__":
    main()

"""Open-source self-hosted ReID inference API.

This service is an optional companion to the browser-only demo. It performs
embedding extraction and vector retrieval, but never makes an identity
assertion or automated decision. Every response is a review aid for an
authorized operator.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised when the optional dependency is absent
    faiss = None


ROOT = Path(os.getenv("REID_ROOT", Path(__file__).resolve().parents[1]))
MODEL_PATH = Path(os.getenv("REID_MODEL", ROOT / "web/public/model/siamese_encoder.onnx"))
GALLERY_PATH = Path(os.getenv("REID_GALLERY", ROOT / "web/public/data/gallery.json"))
VERSION_PATH = Path(os.getenv("REID_VERSION", ROOT / "web/public/data/gallery_version.json"))
MAX_UPLOAD_BYTES = int(os.getenv("REID_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text())


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _load_runtime() -> tuple[ort.InferenceSession, str, str]:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"ONNX model is missing: {MODEL_PATH}")
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name


def _load_gallery() -> tuple[list[dict[str, Any]], np.ndarray, Any, dict[str, Any]]:
    records = _read_json(GALLERY_PATH)
    if not isinstance(records, list) or not records:
        raise RuntimeError("Gallery JSON must contain a non-empty list")
    embeddings = _normalise(np.asarray([record["embedding"] for record in records], dtype=np.float32))
    index = None
    if faiss is not None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    version = _read_json(VERSION_PATH) if VERSION_PATH.exists() else {}
    return records, embeddings, index, version


try:
    SESSION, INPUT_NAME, OUTPUT_NAME = _load_runtime()
    GALLERY, GALLERY_EMBEDDINGS, INDEX, GALLERY_VERSION = _load_gallery()
except Exception as exc:  # expose a health error rather than failing import with an opaque traceback
    SESSION = None
    INPUT_NAME = "images"
    OUTPUT_NAME = "embeddings"
    GALLERY = []
    GALLERY_EMBEDDINGS = np.empty((0, 0), dtype=np.float32)
    INDEX = None
    GALLERY_VERSION = {}
    STARTUP_ERROR = str(exc)
else:
    STARTUP_ERROR = None


app = FastAPI(
    title="Siamese Person ReID Review API",
    version="1.0.0",
    description=(
        "Open-source, self-hostable embedding retrieval for authorized research. "
        "Results are review candidates only and are not identity decisions."
    ),
)

allowed_origins = [origin.strip() for origin in os.getenv("REID_CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_ready() -> None:
    if STARTUP_ERROR or SESSION is None or not GALLERY:
        raise HTTPException(status_code=503, detail=f"ReID service unavailable: {STARTUP_ERROR or 'not loaded'}")


def _image_tensor(payload: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(payload)).convert("RGB").resize((64, 128), Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Upload a valid JPG, PNG, or WEBP image") from exc
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((pixels - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)


def _encode(payload: bytes) -> np.ndarray:
    tensor = _image_tensor(payload)
    output = SESSION.run([OUTPUT_NAME], {INPUT_NAME: tensor})[0]
    return _normalise(np.asarray(output, dtype=np.float32))[0]


def _threshold() -> float | None:
    value = GALLERY_VERSION.get("evaluation", {}).get("threshold_calibration", {}).get("review_threshold")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _rank(query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    limit = min(top_k, len(GALLERY))
    if INDEX is not None:
        scores, indices = INDEX.search(query[None, :].astype(np.float32), limit)
        return [(int(index), float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
    scores = GALLERY_EMBEDDINGS @ query
    indices = np.argsort(-scores)[:limit]
    return [(int(index), float(scores[index])) for index in indices]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if STARTUP_ERROR is None else "error",
        "model_loaded": SESSION is not None,
        "gallery_images": len(GALLERY),
        "embedding_dim": int(GALLERY_EMBEDDINGS.shape[1]) if GALLERY_EMBEDDINGS.ndim == 2 and GALLERY_EMBEDDINGS.size else None,
        "index_backend": "faiss" if INDEX is not None else "numpy-fallback",
        "error": STARTUP_ERROR,
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    _require_ready()
    return {
        "gallery_version": GALLERY_VERSION.get("version"),
        "review_threshold": _threshold(),
        "gallery_images": len(GALLERY),
        "embedding_dim": int(GALLERY_EMBEDDINGS.shape[1]),
        "index_backend": "faiss" if INDEX is not None else "numpy-fallback",
        "human_review_required": True,
    }


@app.post("/search")
async def search(
    image: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    _require_ready()
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")

    query = _encode(payload)
    ranked = _rank(query, top_k)
    threshold = _threshold()
    candidates = []
    for rank, (index, score) in enumerate(ranked, start=1):
        record = GALLERY[index]
        candidates.append(
            {
                "rank": rank,
                "score": round(score, 6),
                "review_label": "REVIEW CANDIDATE" if threshold is not None and score >= threshold else "LOW CONFIDENCE",
                "image": record.get("image"),
                "pid": record.get("pid"),
                "camid": record.get("camid"),
            }
        )

    top_score = candidates[0]["score"] if candidates else None
    review_state = "NO RELIABLE MATCH"
    if top_score is not None and threshold is not None and top_score >= threshold:
        review_state = "REVIEW CANDIDATE"
    return {
        "review_state": review_state,
        "review_threshold": threshold,
        "gallery_version": GALLERY_VERSION.get("version"),
        "human_review_required": True,
        "candidates": candidates,
    }

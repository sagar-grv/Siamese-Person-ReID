# Siamese Person Re-Identification

A compact Siamese-network person re-identification system that learns visual embeddings for cross-camera image retrieval. The project includes a PyTorch training pipeline, held-out evaluation, ONNX export, and a browser demo using ONNX Runtime Web.

> This is a research prototype for visual retrieval, not a biometric identification or surveillance system. Uploaded images are processed locally in the browser and are not sent to a server.

## How it works

Two person images pass through the same CNN encoder with shared weights. Contrastive loss pulls same-identity pairs closer in embedding space and pushes different-identity pairs apart. The trained encoder produces a normalized 128-dimensional vector. During retrieval, the query vector is compared with precomputed gallery vectors using cosine similarity.

| Component | Implementation |
|---|---|
| Encoder | Compact CNN with BatchNorm, ReLU, pooling, and adaptive average pooling |
| Embedding | 128-dimensional L2-normalized vector |
| Loss | Contrastive loss with margin `1.0` |
| Training | AdamW, cosine learning-rate schedule, deterministic seed `42` |
| Inference | ONNX Runtime Web in the browser |
| Deployment | Static Vite application on Vercel |

## Dataset

The project uses a deterministic subset of the public [Market-1501 dataset][1]. Training and evaluation identities are disjoint so that retrieval is tested on identities not used for training.

| Split | Identities | Images | Purpose |
|---|---:|---:|---|
| Train | 100 | 795 | Contrastive pair training |
| Query | 100 | 200 | Held-out retrieval queries |
| Gallery | 100 | 854 | Retrieval candidates |

The dataset images are not committed to Git. The model artifacts and browser gallery included in `web/public` are already prepared for the demo.

## Results

Training configuration: 12 epochs, 1,200 pairs per epoch, batch size 32, embedding size 128, margin 1.0, seed 42.

| Metric | Result |
|---|---:|
| Top-1 accuracy | **58.5%** |
| Top-5 accuracy | **82.5%** |
| Mean average precision | **0.3296** |

These are measurements on the project subset and should not be interpreted as official full-dataset benchmark results.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The dataset archive can be downloaded from the public mirror below:

```bash
mkdir -p datasets
curl -L --fail -o datasets/Market-1501.zip \
  'https://huggingface.co/datasets/tuandunghcmut/MyPublicStorage/resolve/main/Market-1501-v15.09.15.zip?download=true'
unzip -q datasets/Market-1501.zip -d datasets
```

Prepare the deterministic subset under:

```text
data/market1501_subset/
├── train/
├── query/
└── gallery/
```

The training script validates the directories, filenames, and identity split before starting.

## Train and evaluate

```bash
python3 train_siamese.py \
  --data-root data/market1501_subset \
  --epochs 12 \
  --pairs-per-epoch 1200 \
  --batch-size 32
```

Generated artifacts are written to `artifacts/`:

```text
siamese_market1501.pt
siamese_encoder.onnx
metrics.json
history.json
gallery.json
```

To evaluate an existing checkpoint and regenerate exports:

```bash
python3 evaluate_export.py \
  --data-root data/market1501_subset \
  --artifacts artifacts
```

## Run the browser demo

```bash
cd web
pnpm install
pnpm dev
```

The frontend loads the ONNX model, gallery embeddings, sample query, and gallery images from `web/public`. Image preprocessing, embedding extraction, and nearest-neighbor ranking run in the browser.

To test the production bundle locally:

```bash
cd web
pnpm build
pnpm preview --host 0.0.0.0 --port 4173
```

## Demo

Try the deployed browser demo: [Trace / Siamese Re-ID Lab](https://siamese-sg.vercel.app/)

The demo loads the ONNX model and performs image retrieval directly in the browser.

## Architecture

![Siamese person re-identification architecture](docs/architecture.png)

## Repository structure

```text
src/reid_core.py       Siamese encoder, contrastive loss, pairs, metrics
train_siamese.py       Training, evaluation, checkpointing, ONNX export
evaluate_export.py     Checkpoint evaluation and export recovery
web/                   Vite browser application
reports/               Recorded metrics and training history
requirements.txt       Python dependencies
```

## References
```
[1]: https://zheng-lab-anu.github.io/Project/project_reid.html "Market-1501 dataset project page"

[2]: https://www.cs.utoronto.ca/~rsalakhu/papers/oneshot1.pdf "Koch, Zemel, and Salakhutdinov — Siamese Neural Networks for One-shot Image Recognition"

[3]: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf "Hadsell, Chopra, and LeCun — Dimensionality Reduction by Learning an Invariant Mapping"

[4]: https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html "Zheng et al. — Scalable Person Re-identification: A Benchmark"

[5]: https://arxiv.org/abs/1502.02171 "Liao et al. — Person Re-identification Meets Image Search"
```

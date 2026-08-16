# Technical README: Siamese Person Re-Identification

## 1. Project overview

This repository implements a compact **Siamese neural network for person re-identification (ReID)**. The system receives a cropped person image, converts it into a normalized visual embedding, and retrieves the most similar gallery images from other camera views. The central design is metric learning rather than closed-set classification: the model learns a distance function that can generalize to identities that were not present during training.

The implementation is intentionally small enough to train locally and deploy as a static browser application. It uses PyTorch for training, ONNX for model interchange, ONNX Runtime Web for client-side inference, and Vite/Vercel for the web experience. The frontend does not send uploaded images to a backend; preprocessing, encoding, and nearest-neighbor ranking happen in the browser.

> **Scope and safety statement:** This is a research and engineering prototype for visual retrieval. It is not a biometric identification system, does not establish a person’s identity, and should not be used for surveillance, access control, law-enforcement decisions, or other high-impact decisions.

## 2. Why person re-identification needs metric learning

Person re-identification asks whether images captured by different cameras contain the same person. The challenge is that the same identity can change appearance because of viewpoint, illumination, pose, occlusion, camera quality, and background. At the same time, different people can share clothing colors, body shape, or scene context.

A conventional classifier learns a fixed set of training identities. That is not ideal for ReID because the gallery may contain identities that were never observed during training. A Siamese model instead learns an embedding function `f(x)` such that images of the same identity are close and images of different identities are separated by a margin. Once the embedding function is trained, retrieval can be performed against a new gallery without retraining the classifier.

The method follows the Siamese metric-learning formulation introduced for one-shot image recognition [1] and uses the margin-based contrastive objective described by Hadsell, Chopra, and LeCun [5]. The application domain is based on the Market-1501 person ReID benchmark [2].

## 3. Conceptual model

The two input images are processed by the **same encoder with shared parameters**. The repository keeps the compact contrastive Siamese pipeline in `train_siamese.py` and adds a stronger production/demo pipeline in `train_upgraded.py`. The upgraded path uses a pretrained ResNet-18 backbone, a BatchNorm neck, a 256-dimensional projection, identity classification, and batch-hard triplet mining.

```mermaid
flowchart LR
    A[Person crop A] --> PA[Resize 128 x 64<br/>augment + normalize]
    B[Person crop B] --> PB[Resize 128 x 64<br/>augment + normalize]
    PA --> E[Shared ResNet-18<br/>pretrained backbone]
    PB --> E
    E --> BN[BatchNorm neck]
    BN --> Z[256D normalized embedding]
    BN --> C[Identity classifier]
    Z --> H[Batch-hard triplet mining]
    C --> CE[Cross-entropy + label smoothing]
    H --> U[AdamW update]
    CE --> U
```

During inference, only one image is encoded at a time. The gallery is encoded offline once, and the query embedding is compared with the stored gallery embeddings using cosine similarity. Because the model outputs L2-normalized vectors, cosine similarity and Euclidean distance are directly related:

```text
||z||₂ = 1
cosine_similarity(zq, zg) = zq · zg
squared_euclidean_distance(zq, zg) = 2 - 2(zq · zg)
```

## 4. Mathematical formulation

Let `fθ(x)` denote the encoder with parameters `θ`, and let `y` be the pair label. In this repository, `y = 0` means that the two images depict the same identity and `y = 1` means that they depict different identities.

The model first produces normalized embeddings:

```text
z₁ = normalize(fθ(x₁))
z₂ = normalize(fθ(x₂))
D = ||z₁ - z₂||₂
```

The implemented contrastive loss is:

```text
L(x₁, x₂, y) = (1 - y) D² + y max(m - D, 0)²
```

where `m` is the margin. The implementation uses `m = 1.0`. For a positive pair, the loss penalizes distance directly. For a negative pair, the loss is zero once the embeddings are at least one margin unit apart. This encourages the network to learn an identity-sensitive geometry without requiring a classifier head over all possible people.

The exact code is in [`src/reid_core.py`](src/reid_core.py), specifically `SiameseEncoder`, `ContrastiveLoss`, and `PairDataset`.

## 5. Encoder architecture

The browser demo now uses the stronger encoder produced by `train_upgraded.py`. It retains the same fixed `128 × 64` input contract as the compact Siamese model, so the frontend preprocessing and ONNX interface remain unchanged.

| Stage | Operation | Output |
|---|---|---:|
| Input | RGB image resized to `128 × 64` | 3 channels |
| Backbone | ImageNet-pretrained ResNet-18 | 512 pooled features |
| Metric neck | BatchNorm1d with frozen bias | 512 features |
| Projection | Linear layer + BatchNorm1d | 256 features |
| Retrieval output | L2 normalization | 256-dimensional unit vector |
| Classification branch | Linear identity classifier | 751 training classes |

The normalized projection is used for retrieval, while the classifier operates on the neck features. Separating these spaces follows the strong-baseline motivation that classification and metric objectives can have different feature preferences [6].

## 6. Input preprocessing and augmentation

Every image is converted to RGB and normalized with ImageNet channel statistics. Training uses a resize to `144 × 72`, a random `128 × 64` crop, horizontal flipping, moderate color jitter, and random erasing. Evaluation and browser inference use deterministic resize only. The random erasing step simulates partial occlusion without changing the model’s input contract.

The browser implementation mirrors the final deterministic preprocessing in `web/src/main.js`: RGB channel order, `128 × 64` dimensions, float32 tensors, and the same mean/std values. This alignment is essential because preprocessing drift can reduce retrieval quality even when the ONNX model loads successfully.

## 7. Dataset: Market-1501

Market-1501 is a public person ReID benchmark collected across six cameras. The official dataset page reports 1,501 identities and 32,668 annotated pedestrian bounding boxes, with separate training, query, gallery, and ground-truth components [2]. The original archive is intentionally not committed to this repository because it contains thousands of image files.

For a fast reproducible experiment, this project uses a deterministic subset prepared from the public Market-1501 archive. Training identities and evaluation identities are disjoint. This is important because the purpose of ReID metric learning is to generalize the learned visual similarity function to identities not used for parameter updates.

| Split | Identities | Images | Use |
|---|---:|---:|---|
| `train` | 100 | 795 | Positive and negative pair sampling |
| `query` | 100 | 200 | Held-out retrieval queries |
| `gallery` | 100 | 854 | Held-out retrieval candidates |

The subset uses the Market-1501 filename convention `<person_id>_c<camera_id>...jpg`. The parser extracts the identity and camera identifiers from each filename. A training assertion verifies that no identity appears in both the training and evaluation sets.

### Dataset acquisition

The official benchmark page should be treated as the primary dataset reference [2]. The experiment archive was downloaded from the following public mirror:

```text
https://huggingface.co/datasets/tuandunghcmut/MyPublicStorage/resolve/main/Market-1501-v15.09.15.zip?download=true
```

The archive can be downloaded and extracted with:

```bash
mkdir -p /home/ubuntu/datasets
curl -L --fail -o /home/ubuntu/datasets/Market-1501-v15.09.15.zip \
  'https://huggingface.co/datasets/tuandunghcmut/MyPublicStorage/resolve/main/Market-1501-v15.09.15.zip?download=true'
unzip -q /home/ubuntu/datasets/Market-1501-v15.09.15.zip \
  -d /home/ubuntu/datasets
```

The original subset preparation selected the first 100 identities available in `bounding_box_train`, then selected the first 100 query identities not present in the training set. It copied bounded numbers of images into `data/market1501_subset/train`, `data/market1501_subset/query`, and `data/market1501_subset/gallery`. The downloaded dataset and generated local subset are ignored by Git.

## 8. Identity-balanced batches and hard negatives

The upgraded pipeline replaces random pair generation with identity-balanced batches. Each batch samples 8 identities and 4 images per identity. This creates positive and negative relationships inside every batch and allows the loss to choose the hardest positive and hardest negative for each anchor.

For normalized embeddings, the pairwise distance is computed from cosine similarity. The batch-hard triplet objective is:

```text
L_triplet = max(d_hard_positive - d_hard_negative + margin, 0)
```

The training objective combines this metric loss with identity classification. The original `PairDataset` remains available in `src/reid_core.py` for the compact paper-faithful baseline.

## 9. Training configuration

The compact baseline is trained by [`train_siamese.py`](train_siamese.py). The stronger full-split model is trained by [`train_upgraded.py`](train_upgraded.py). The upgraded entry point validates the Market-1501 identity split, creates identity-balanced batches, trains the joint metric/classification objective, saves the best training-loss checkpoint, evaluates the official query/gallery folders, writes metrics, and exports browser assets from the demo subset.

| Parameter | Upgraded value |
|---|---:|
| Epochs | 12 |
| Batches per epoch | 150 |
| Identities per batch | 8 |
| Images per identity | 4 |
| Embedding dimension | 256 |
| Batch-hard margin | 0.3 |
| Classification loss | Cross-entropy, label smoothing 0.1 |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0005 |
| Optimizer | AdamW |
| Schedule | 2-epoch warmup + cosine decay |
| Gradient clipping | 5.0 |
| Random seed | 42 |
| Backbone | ImageNet-pretrained ResNet-18 |

Run the upgraded pipeline with:

```bash
python3 train_upgraded.py \
  --market-root datasets/Market-1501-v15.09.15 \
  --demo-root data/market1501_subset \
  --artifacts artifacts/upgraded \
  --epochs 12 \
  --batches-per-epoch 150 \
  --identities-per-batch 8 \
  --images-per-identity 4
```

The generated local artifacts are:

```text
artifacts/
├── siamese_market1501.pt       # PyTorch checkpoint
├── siamese_encoder.onnx        # self-contained browser model
├── gallery.json                # gallery metadata + embeddings
├── metrics.json                # retrieval metrics
└── history.json                # epoch-level training history
```

## 10. Evaluation protocol and metrics

The upgraded evaluation encodes every query and gallery image, computes cosine similarities, removes gallery images with the same identity and camera as the query, ranks the remaining gallery, and reports CMC-style top-k accuracy and mAP. This same-camera filtering makes the full-split evaluation closer to the standard Market-1501 protocol than the original smoke-test metrics.

### Top-1 and Top-5 accuracy

For each query, Top-1 is one when the highest-ranked gallery image has the same identity. Top-5 is one when at least one of the five highest-ranked gallery images has the same identity. The reported score is the mean over all queries.

### Mean average precision

For each query, the ranked gallery is converted into a binary relevance vector. Precision is computed at each rank where a relevant image occurs, and those precision values are averaged over all relevant images for that query. The final mAP is the mean of query-level average precisions.

### Measured results

The upgraded run used seed 42, 12 epochs, 150 batches per epoch, and the full Market-1501 training split.

| Full evaluation metric | Value |
|---|---:|
| Queries | 3,368 |
| Gallery images | 19,732 |
| Training images | 12,936 |
| Training identities | 751 |
| Top-1 accuracy | **47.9%** |
| Top-5 accuracy | **72.0%** |
| Mean average precision | **0.2773** |

On the 200-query / 854-gallery browser subset, using the same camera-filtered protocol, the compact baseline scored 13.0% top-1 and 0.1848 mAP, while the upgraded model scored 62.0% top-1 and 0.5037 mAP. These subset results measure the implementation change; the full-split numbers are the more realistic benchmark protocol. The source files are [`reports/accuracy_comparison.json`](reports/accuracy_comparison.json), [`reports/upgraded_full_metrics.json`](reports/upgraded_full_metrics.json), and [`reports/upgraded_demo_metrics.json`](reports/upgraded_demo_metrics.json).

## 11. ONNX export and browser inference

The PyTorch encoder is exported to ONNX with a dynamic batch dimension, input name `images`, output name `embeddings`, opset 18, and `external_data=False`. A self-contained file is used because ONNX Runtime Web cannot rely on the mounted external-data mechanism used by some ONNX exports.

The frontend loads the model using `onnxruntime-web`. The application performs the following sequence:

```text
1. Read an uploaded image locally with FileReader.
2. Draw it into a 64 × 128 canvas.
3. Apply the same RGB normalization as the Python pipeline.
4. Create a [1, 3, 128, 64] float32 tensor.
5. Run the ONNX model in the browser.
6. Normalize the query embedding.
7. Compute dot products against precomputed gallery embeddings.
8. Sort by similarity and render the top ten results.
```

The static assets are located under `web/public`:

```text
web/public/
├── model/siamese_encoder.onnx
├── data/gallery.json
├── data/demo-query.jpg
└── gallery/*.jpg
```

The gallery JSON contains the filename, person identifier, camera identifier, public image path, and rounded embedding vector for each gallery image. The deployment does not need a Python server, GPU, API key, database, or image-upload endpoint.

## 12. Web application structure

The frontend is a Vite application with a deliberately small runtime surface.

| File | Responsibility |
|---|---|
| `web/index.html` | Semantic page structure and accessibility labels |
| `web/src/main.js` | ONNX session loading, preprocessing, inference, ranking, and UI state |
| `web/src/style.css` | Responsive Trace/Lab visual system, panels, results grid, and motion |
| `web/public/model/siamese_encoder.onnx` | Browser inference model |
| `web/public/data/gallery.json` | Precomputed retrieval index |
| `web/public/gallery/` | Static gallery images |
| `web/vercel.json` | Configuration when Vercel Root Directory is `web` |

The project deliberately uses a browser-only architecture for the public demo. The upgraded ONNX export is larger than the compact baseline because it contains the ResNet-18 backbone, but it keeps uploaded images local to the user’s browser and avoids deploying a heavyweight Python runtime to a serverless function.

## 13. Local web development

From the repository root:

```bash
cd web
pnpm install
pnpm dev
```

For a production-style local check:

```bash
cd web
pnpm build
pnpm preview --host 0.0.0.0 --port 4173
```

The production build should contain `dist/index.html`, the bundled JavaScript and CSS, the ONNX Runtime Web WASM asset, the ONNX model, gallery JSON, sample query, and gallery images.

## 14. Vercel deployment

When importing the GitHub repository into Vercel, use the `web` directory as the project Root Directory. The nested `web/vercel.json` is configured for this arrangement:

| Vercel setting | Value |
|---|---|
| Root Directory | `web` |
| Framework Preset | `Vite` |
| Install Command | `npm install --no-audit --no-fund` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Environment variables | None |

The root-level `vercel.json` remains available for deployments configured from the repository root, but the most reliable dashboard configuration is Root Directory `web` with commands that do not contain `cd web`. The distinction matters because Vercel executes commands relative to the selected Root Directory.

## 15. Reproducibility checklist

A reproducible run should use the same dataset subset, seed, input dimensions, normalization constants, pair-generation policy, embedding dimension, margin, optimizer, and training schedule. The following checklist captures the required conditions:

| Check | Expected value |
|---|---|
| Dataset identities | Train and evaluation identities are disjoint |
| Seed | 42 |
| Image tensor | `float32`, shape `3 × 128 × 64` |
| Normalization | Mean `[0.485, 0.456, 0.406]`; std `[0.229, 0.224, 0.225]` |
| Positive label | 0.0 |
| Negative label | 1.0 |
| Margin | 1.0 |
| Embedding | L2-normalized, 256 dimensions |
| Gallery score | Query/gallery dot product |
| Browser preprocessing | Matches Python preprocessing |

For stronger scientific reporting, repeat the experiment across several seeds, retain a validation split, report confidence intervals, apply the official Market-1501 evaluation protocol, and compare against stronger ReID backbones such as OSNet or ResNet-based baselines.

## 16. Known limitations

The repository still includes the compact subset-trained Siamese baseline for comparison, but the browser demo now uses the upgraded ResNet-18 checkpoint trained on the full Market-1501 training split. The upgraded pipeline uses same-camera filtering and batch-hard mining, but it is still a small research prototype rather than a tuned state-of-the-art system. The browser gallery is a fixed precomputed index, so adding new gallery images requires regenerating embeddings and redeploying the static assets.

The system should not be interpreted as robust identity recognition in uncontrolled real-world environments. Performance may degrade under different camera domains, clothing changes, occlusion, crowding, unusual poses, image compression, or demographic distribution shifts. The public demo displays dataset identifiers for technical inspection; a production system should use privacy-preserving identifiers and appropriate governance.

## 17. Future work

The most useful next steps are validation-based checkpoint selection, multi-seed reporting, cross-domain testing, camera-aware sampling, confidence calibration, query expansion or re-ranking, OSNet/ResNet comparisons, and a lighter mobile export. A production deployment should also define data-retention rules, user consent, access control, audit logging, and an explicit prohibition on high-impact biometric use.

## References

[1]: https://arxiv.org/abs/1503.03832 "Siamese Neural Networks for One-shot Image Recognition"

[2]: https://zheng-lab-anu.github.io/Project/project_reid.html "Market-1501 Dataset official project page"

[3]: https://github.com/going-doer/Paper2Code "Paper2Code repository"

[4]: https://vercel.com/docs/deployments "Vercel deployment documentation"

[5]: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf "Dimensionality Reduction by Learning an Invariant Mapping"

[6]: https://arxiv.org/abs/1906.08332 "A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification"

[7]: https://arxiv.org/abs/2102.04378 "TransReID: Transformer-based Object Re-Identification"

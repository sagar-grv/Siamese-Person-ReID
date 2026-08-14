# Trace / Siamese Person Re-Identification Lab

A reproducible **Siamese-network person re-identification prototype** rebuilt around the original metric-learning idea: two image crops pass through the same encoder, and contrastive learning places same-identity views near one another while separating different identities. The repository includes a trainable PyTorch pipeline, held-out retrieval evaluation, a self-contained ONNX export, and a browser-native demo intended for Vercel.

> **Important:** This is a research and engineering prototype for visual retrieval, not a biometric identification service. The demo runs inference in the browser and does not upload the selected image to a server.

## What changed

The original checkout depended on an uncommitted image directory, missing model artifacts, and a Streamlit runtime that was not deployable as a lightweight Vercel site. The rebuild makes the data contract explicit, trains a compact model on a documented public subset, evaluates on identities excluded from training, and ships the final inference experience as a static frontend.

| Layer | Implementation |
|---|---|
| Metric-learning core | Shared-weight CNN encoder with L2-normalized 128-dimensional embeddings |
| Objective | Contrastive loss with margin 1.0; label 0 means same identity and label 1 means different identity |
| Dataset | Market-1501 public benchmark; deterministic 100-identity train subset and 100 identity-disjoint evaluation identities |
| Evaluation | Top-1 accuracy, Top-5 accuracy, and mean average precision over query-to-gallery retrieval |
| Deployment model | ONNX Runtime Web in a Vite static site; precomputed gallery embeddings and local gallery images |
| Hosting target | Vercel static deployment from the `web/` build output |

## Paper2Code-inspired workflow

The rebuild follows the practical structure emphasized by [Paper2Code][3]: first record the plan and assumptions, then separate analysis artifacts from implementation, and finally produce a runnable repository. The corresponding artifacts in this project are `research_notes.md`, `src/reid_core.py`, `train_siamese.py`, `evaluate_export.py`, `artifacts/metrics.json`, and the final `web/` application.

## Dataset and evaluation protocol

The source benchmark is **Market-1501**, a six-camera person re-identification dataset with 1,501 identities and 32,668 annotated pedestrian bounding boxes. Its official page describes separate training, query, gallery, and ground-truth components [2]. To make local training practical and reproducible, the included experiment uses the following deterministic subset:

| Split | Identities | Images | Purpose |
|---|---:|---:|---|
| Training | 100 | 795 | Pair sampling for contrastive training |
| Query | 100 | 200 | Held-out retrieval queries |
| Gallery | 100 | 854 | Held-out retrieval gallery |

Training and evaluation identities are disjoint. The dataset archive is not committed to Git; the download and subset preparation commands are recorded below. The public archive used for this run was obtained from a Hugging Face mirror of the Market-1501 ZIP, while the official dataset page remains the authoritative benchmark reference [2].

## Measured result

The model was trained for 12 epochs with seed 42, 1,200 generated pairs per epoch, batch size 32, and a 128-dimensional embedding. The resulting checkpoint was evaluated against all 200 held-out queries and 854 gallery images.

| Metric | Result |
|---|---:|
| Top-1 retrieval accuracy | **58.5%** |
| Top-5 retrieval accuracy | **82.5%** |
| Mean average precision | **0.3296** |
| Browser inference latency in local preview | **48 ms** for the default query |

These figures are subset-prototype measurements, not claims of state-of-the-art Market-1501 performance. The complete metrics are saved in `artifacts/metrics.json` during training and are represented in the deployed demo through the precomputed gallery index.

## Local setup

The Python training environment requires PyTorch, scikit-learn, Pillow, Matplotlib, ONNX, ONNX Runtime, and ONNX Script. Install them with:

```bash
sudo pip3 install torch torchvision scikit-learn pillow matplotlib onnx onnxruntime onnxscript tqdm
```

Download the public archive and extract it:

```bash
mkdir -p /home/ubuntu/datasets
curl -L --fail -o /home/ubuntu/datasets/Market-1501-v15.09.15.zip \
  'https://huggingface.co/datasets/tuandunghcmut/MyPublicStorage/resolve/main/Market-1501-v15.09.15.zip?download=true'
unzip -q /home/ubuntu/datasets/Market-1501-v15.09.15.zip -d /home/ubuntu/datasets
```

The deterministic subset can be prepared with the same logic used for this run: choose 100 available identities from `bounding_box_train`, then choose the first 100 query identities not present in training, copying bounded numbers of images into `data/market1501_subset/{train,query,gallery}`. The resulting directory is intentionally ignored by Git because it contains dataset images.

Train and export the model:

```bash
python3 train_siamese.py \
  --data-root data/market1501_subset \
  --epochs 12 \
  --pairs-per-epoch 1200 \
  --batch-size 32
```

If a checkpoint already exists and only evaluation or export must be repeated, run:

```bash
python3 evaluate_export.py \
  --data-root data/market1501_subset \
  --artifacts artifacts
```

The export uses `external_data=False` so the browser receives one self-contained `siamese_encoder.onnx` file. This avoids the mounted-file dependency that caused the initial browser preview failure.

## Run the web demo

```bash
cd web
pnpm install
pnpm dev
```

The frontend loads `/model/siamese_encoder.onnx`, `/data/gallery.json`, `/data/demo-query.jpg`, and the gallery JPEGs from static assets. Uploads are converted to tensors locally in the browser, encoded with ONNX Runtime Web, and ranked by cosine similarity against the precomputed gallery embeddings.

To reproduce the production bundle locally:

```bash
cd web
pnpm build
pnpm preview --host 0.0.0.0 --port 4173
```

## Deploy to Vercel

The repository includes `vercel.json` configured to build `web/` and publish `web/dist`. With the Vercel CLI authenticated:

```bash
npx vercel login
npx vercel --prod
```

Alternatively, import the GitHub repository in the Vercel dashboard and keep the repository-root configuration from `vercel.json`. No model API key or server-side secret is required because inference is client-side.

## Repository map

```text
src/reid_core.py       Shared encoder, contrastive loss, pair sampling, metrics
train_siamese.py       Reproducible training and export entry point
evaluate_export.py     Re-evaluate checkpoint and regenerate web artifacts
artifacts/             Local checkpoint, metrics, history, and gallery index
web/index.html         Trace/Lab interface
web/src/main.js        Browser preprocessing, ONNX inference, retrieval UI
web/src/style.css      Visual system and responsive layout
web/public/model/      Self-contained ONNX model for the demo
web/public/data/       Gallery index and default query image
web/public/gallery/    Static held-out gallery image set
```

## References

[1]: https://arxiv.org/abs/1503.03832 "Siamese Neural Networks for One-shot Image Recognition"

[2]: https://zheng-lab-anu.github.io/Project/project_reid.html "Market-1501 Dataset official project page"

[3]: https://github.com/going-doer/Paper2Code "Paper2Code repository"

[4]: https://vercel.com/docs/deployments "Vercel deployment documentation"

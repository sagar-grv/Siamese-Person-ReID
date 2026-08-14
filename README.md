# 🔍 Person Re-Identification System

A Siamese Neural Network-based person re-identification system comparing **Triplet Loss** vs **Contrastive Loss**.

## 📁 Project Structure

```
Siemese_ATML/
├── data/                           # Dataset
│   └── SNN-TL-Data/
│       ├── train/                  # Training images
│       └── train.csv               # Triplet annotations
├── models/                         # Trained models
│   ├── triplet/                    # Triplet Loss model
│   │   ├── best_model.pt
│   │   └── database.csv
│   └── contrastive/                # Contrastive Loss model
│       ├── best_model_contrastive.pt
│       └── database_contrastive.csv
├── outputs/                        # Generated outputs
│   ├── training_history.png
│   ├── training_history_contrastive.png
│   ├── triplet_sample.png
│   └── results.png
├── src/                            # Training scripts
│   ├── train_triplet.py
│   └── train_contrastive.py
├── archive/                        # Legacy files
├── app.py                          # Streamlit web app
├── requirements.txt                # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the Dataset

The image files and trained model artifacts are intentionally excluded from Git because they are large. The repository includes the 4,000-row annotation file at `data/SNN-TL-Data/train.csv`, but you must place the corresponding images in:

```text
data/SNN-TL-Data/train/
```

Each row in `train.csv` must resolve its `Anchor`, `Positive`, and `Negative` filenames inside that directory. The training scripts validate this requirement before constructing a DataLoader and report the missing paths directly.

### 3. Train a Model

```bash
python src/train_triplet.py
# or
python src/train_contrastive.py
```

Training creates the required `models/` and `outputs/` directories automatically. Generated checkpoints and embedding databases are not committed to Git.

### 4. Run Streamlit App

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser. The app shows which model or embedding database is missing if training has not been completed.

## 🧠 Models

| Loss Function | Description | Model File |
|---------------|-------------|------------|
| **Triplet Loss** | Uses anchor-positive-negative triplets | `models/triplet/best_model.pt` |
| **Contrastive Loss** | Uses positive-negative pairs | `models/contrastive/best_model_contrastive.pt` |

Both models use **EfficientNet-B0** backbone with 512-dimensional embeddings.

## 📊 Training

### Train Triplet Loss Model

```bash
python src/train_triplet.py
```

### Train Contrastive Loss Model

```bash
python src/train_contrastive.py
```

## 🖥️ Features

- **Model Selection**: Switch between Triplet and Contrastive loss models
- **Compare Mode**: View results from both models side-by-side
- **GPU Accelerated**: Automatic CUDA support with mixed precision (AMP)
- **Interactive UI**: Upload images and find matching persons

## ⚙️ Configuration

Training parameters can be modified in `src/train_*.py`:

- `BATCH_SIZE`: 64 (default)
- `EPOCHS`: 10
- `LR`: 0.001
- `EARLY_STOP_PATIENCE`: 3

## 🛠️ Troubleshooting

If a training command exits before loading data, check that the dataset directory exists and that all filenames in `train.csv` are present. The scripts report the number of missing images and examples of the missing filenames instead of failing later inside a worker process.

## 📜 License
MIT License

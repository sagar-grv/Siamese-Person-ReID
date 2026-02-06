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

### 2. Run Streamlit App

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

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

## 📜 License

MIT License

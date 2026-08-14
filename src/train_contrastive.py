# -*- coding: utf-8 -*-
"""
Siamese Neural Network with Contrastive Loss for Person Re-Identification
Adapted for local execution with GPU support
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import timm

import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from skimage import io
from sklearn.model_selection import train_test_split

from tqdm import tqdm

# Import shared validation helpers when executed as `python src/train_contrastive.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import ensure_directories, load_validated_triplets

# ==================== Configuration ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up from src/ to project root

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'SNN-TL-Data', 'train')
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'SNN-TL-Data', 'train.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'contrastive')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10
EARLY_STOP_PATIENCE = 3
NUM_WORKERS = 4
MARGIN = 1.0  # Contrastive loss margin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print(f"\n{'='*60}")
print(f"🚀 DEVICE: {DEVICE}")
print(f"📊 LOSS: Contrastive Loss (margin={MARGIN})")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}")
print(f"{'='*60}\n")


# ==================== Contrastive Loss ====================
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Contrastive loss formula
        # label = 0 means similar (positive pair), label = 1 means dissimilar (negative pair)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


# ==================== Pair Dataset ====================
class PairDataset(Dataset):
    """Dataset for Contrastive Learning with positive and negative pairs"""
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.df) * 2  # Both positive and negative pairs
    
    def __getitem__(self, index):
        row_idx = index // 2
        is_positive = (index % 2 == 0)  # Alternate between positive and negative
        
        row = self.df.iloc[row_idx]
        
        # Load anchor image
        img1 = io.imread(os.path.join(self.data_dir, row.Anchor))
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        
        if is_positive:
            # Positive pair (same person) - label = 0
            img2 = io.imread(os.path.join(self.data_dir, row.Positive))
            label = 0.0
        else:
            # Negative pair (different person) - label = 1
            img2 = io.imread(os.path.join(self.data_dir, row.Negative))
            label = 1.0
        
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ==================== Model ====================
class SiameseModel(nn.Module):
    """Siamese Network using EfficientNet-B0 backbone"""
    def __init__(self, emb_size=512):
        super(SiameseModel, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(
            in_features=self.efficientnet.classifier.in_features,
            out_features=emb_size
        )
        
    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings


# ==================== Training Functions ====================
def train_fn(model, dataloader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    
    for img1, img2, label in tqdm(dataloader, desc="Training", leave=False):
        img1 = img1.to(DEVICE, non_blocking=True)
        img2 = img2.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                emb1 = model(img1)
                emb2 = model(img2)
                loss = criterion(emb1, emb2, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            emb1 = model(img1)
            emb2 = model(img2)
            loss = criterion(emb1, emb2, label)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def eval_fn(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Validation", leave=False):
            img1 = img1.to(DEVICE, non_blocking=True)
            img2 = img2.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    emb1 = model(img1)
                    emb2 = model(img2)
                    loss = criterion(emb1, emb2, label)
            else:
                emb1 = model(img1)
                emb2 = model(img2)
                loss = criterion(emb1, emb2, label)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def get_encoding_csv(model, anc_img_names, data_dir):
    """Generate embeddings for all anchor images"""
    anc_img_names_arr = np.array(anc_img_names)
    encodings = []
    
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(anc_img_names_arr, desc="Generating embeddings"):
            A = io.imread(os.path.join(data_dir, i))
            A = torch.from_numpy(A).permute(2, 0, 1).float() / 255.0
            A = A.to(DEVICE)
            A_enc = model(A.unsqueeze(0))
            encodings.append(A_enc.squeeze().cpu().detach().numpy())
    
    encodings = np.array(encodings)
    encodings = pd.DataFrame(encodings)
    
    df_enc = pd.concat(
        [anc_img_names.reset_index(drop=True), encodings],
        axis=1
    )
    
    return df_enc


# ==================== Main Execution ====================
def main():
    print("📂 Loading dataset...")
    print(f"   Data directory: {DATA_DIR}")
    print(f"   CSV file: {CSV_FILE}")
    
    try:
        df = load_validated_triplets(CSV_FILE, DATA_DIR)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ ERROR: {exc}")
        return

    ensure_directories((MODEL_DIR, OUTPUT_DIR))
    print(f"   Total triplets: {len(df)}")
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    
    # Create pair datasets
    trainset = PairDataset(train_df, DATA_DIR)
    valset = PairDataset(val_df, DATA_DIR)
    
    print(f"   Training pairs: {len(trainset)}")
    print(f"   Validation pairs: {len(valset)}")
    
    train_loader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        valset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    print(f"\n   Batches in train_loader: {len(train_loader)}")
    print(f"   Batches in val_loader: {len(val_loader)}")
    
    # Initialize model
    print("\n🧠 Initializing Siamese Network (EfficientNet-B0 backbone)...")
    model = SiameseModel()
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    
    # Training loop
    print(f"\n🏋️ Starting training for {EPOCHS} epoch(s)...")
    print(f"   Early stopping patience: {EARLY_STOP_PATIENCE} epochs")
    best_valid_loss = np.inf
    epochs_without_improvement = 0
    train_losses = []
    valid_losses = []
    
    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, scaler)
        valid_loss = eval_fn(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if valid_loss < best_valid_loss:
            model_path = os.path.join(MODEL_DIR, 'best_model_contrastive.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'best_valid_loss': valid_loss,
                'loss_type': 'contrastive',
                'margin': MARGIN
            }, model_path)
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            print(f'   ✅ EPOCH {epoch+1}/{EPOCHS} | train: {train_loss:.4f} | valid: {valid_loss:.4f} | lr: {current_lr:.6f} | SAVED!')
        else:
            epochs_without_improvement += 1
            print(f'   ⏳ EPOCH {epoch+1}/{EPOCHS} | train: {train_loss:.4f} | valid: {valid_loss:.4f} | lr: {current_lr:.6f} | No improvement ({epochs_without_improvement}/{EARLY_STOP_PATIENCE})')
        
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs!")
            break
    
    # Save training history
    print("\n📈 Saving training history plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(valid_losses, label='Valid Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History (Contrastive Loss)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_contrastive.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate embeddings database
    print("\n📊 Generating embeddings database...")
    model_path = os.path.join(MODEL_DIR, 'best_model_contrastive.pt')
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded best model from epoch {checkpoint['epoch']} (valid_loss: {checkpoint['valid_loss']:.4f})")
    
    df_enc = get_encoding_csv(model, df['Anchor'], DATA_DIR)
    
    csv_path = os.path.join(MODEL_DIR, 'database_contrastive.csv')
    df_enc.to_csv(csv_path, index=False)
    print(f"   Embeddings saved to: {csv_path}")
    
    print("\n✅ Contrastive Loss Training Complete!")
    print(f"   Model: best_model_contrastive.pt")
    print(f"   Embeddings: database_contrastive.csv")
    print(f"   History: training_history_contrastive.png")


if __name__ == "__main__":
    main()

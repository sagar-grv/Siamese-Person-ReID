# -*- coding: utf-8 -*-
"""
Siamese Neural Network with Triplet Loss for Person Re-Identification
Adapted for local execution with GPU support
Original: Google Colab - C209_ATML_LAB2.ipynb
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

# Import shared validation helpers when executed as `python src/train_triplet.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import ensure_directories, load_validated_triplets

# ==================== Configuration ====================
# Local paths (relative to project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up from src/ to project root

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'SNN-TL-Data', 'train')
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'SNN-TL-Data', 'train.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'triplet')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

BATCH_SIZE = 64  # Increased for better GPU utilization
LR = 0.001
EPOCHS = 10  # Train for more epochs
EARLY_STOP_PATIENCE = 3  # Stop if no improvement for 3 epochs
NUM_WORKERS = 4  # Parallel data loading

# Device configuration - use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # Mixed precision for faster training

print(f"\n{'='*60}")
print(f"🚀 DEVICE: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}")
print(f"{'='*60}\n")

# ==================== Dataset ====================
class APN_Dataset(Dataset):
    """Anchor-Positive-Negative Dataset for Triplet Learning"""
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # Load images
        A_img = io.imread(os.path.join(self.data_dir, row.Anchor))
        P_img = io.imread(os.path.join(self.data_dir, row.Positive))
        N_img = io.imread(os.path.join(self.data_dir, row.Negative))
        
        # Convert to tensor and normalize to [0, 1]
        A_img = torch.from_numpy(A_img).permute(2, 0, 1).float() / 255.0
        P_img = torch.from_numpy(P_img).permute(2, 0, 1).float() / 255.0
        N_img = torch.from_numpy(N_img).permute(2, 0, 1).float() / 255.0
        
        return A_img, P_img, N_img


# ==================== Model ====================
class APN_Model(nn.Module):
    """Siamese Network using EfficientNet-B0 backbone"""
    def __init__(self, emb_size=512):
        super(APN_Model, self).__init__()
        
        # Use EfficientNet-B0 as backbone (pretrained on ImageNet)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        
        # Replace classifier with embedding layer
        self.efficientnet.classifier = nn.Linear(
            in_features=self.efficientnet.classifier.in_features,
            out_features=emb_size
        )
        
    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings


# ==================== Training Functions ====================
def train_fn(model, dataloader, optimizer, criterion, scaler=None):
    """Training function for one epoch with optional AMP"""
    model.train()
    total_loss = 0.0
    
    for A, P, N in tqdm(dataloader, desc="Training", leave=False):
        A, P, N = A.to(DEVICE, non_blocking=True), P.to(DEVICE, non_blocking=True), N.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if scaler is not None:  # Mixed precision training
            with torch.amp.autocast('cuda'):
                A_embs = model(A)
                P_embs = model(P)
                N_embs = model(N)
                loss = criterion(A_embs, P_embs, N_embs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            A_embs = model(A)
            P_embs = model(P)
            N_embs = model(N)
            loss = criterion(A_embs, P_embs, N_embs)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def eval_fn(model, dataloader, criterion):
    """Evaluation function with AMP support"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for A, P, N in tqdm(dataloader, desc="Validation", leave=False):
            A, P, N = A.to(DEVICE, non_blocking=True), P.to(DEVICE, non_blocking=True), N.to(DEVICE, non_blocking=True)
            
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    A_embs = model(A)
                    P_embs = model(P)
                    N_embs = model(N)
                    loss = criterion(A_embs, P_embs, N_embs)
            else:
                A_embs = model(A)
                P_embs = model(P)
                N_embs = model(N)
                loss = criterion(A_embs, P_embs, N_embs)
            
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
            
            # (C, H, W) -> (1, C, H, W)
            A_enc = model(A.unsqueeze(0))
            
            encodings.append(
                A_enc.squeeze().cpu().detach().numpy()
            )
    
    encodings = np.array(encodings)
    encodings = pd.DataFrame(encodings)
    
    df_enc = pd.concat(
        [anc_img_names.reset_index(drop=True), encodings],
        axis=1
    )
    
    return df_enc


def euclidean_dist(img_enc, anc_enc_arr):
    """Compute Euclidean distance between embeddings"""
    dist = np.sqrt(
        np.dot(img_enc - anc_enc_arr, (img_enc - anc_enc_arr).T)
    )
    return dist


def plot_closest_imgs_simple(anc_img_names, data_dir, img_path, closest_idx, distance, no_of_closest=5):
    """Simple visualization of closest images"""
    fig, axes = plt.subplots(1, no_of_closest + 1, figsize=(15, 4))
    
    # Query image
    query_img = io.imread(img_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image', fontsize=10)
    axes[0].axis('off')
    
    # Closest images
    for i in range(no_of_closest):
        idx = closest_idx[i]
        img_name = anc_img_names.iloc[idx]
        img = io.imread(os.path.join(data_dir, img_name))
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f'Dist: {distance[idx]:.4f}', fontsize=9)
        axes[i + 1].axis('off')
    
    plt.suptitle('Person Re-Identification: Query and Top Matches', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results.png'), dpi=150, bbox_inches='tight')
    plt.show()


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
    print(f"\nSample data:\n{df.head()}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\n   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    
    # Create datasets
    trainset = APN_Dataset(train_df, DATA_DIR)
    valset = APN_Dataset(val_df, DATA_DIR)
    
    # Visualize a sample triplet
    print("\n📸 Visualizing sample triplet...")
    idx = 1
    A, P, N = trainset[idx]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.set_title('Anchor')
    ax1.imshow(A.permute(1, 2, 0))
    ax2.set_title('Positive (Same person)')
    ax2.imshow(P.permute(1, 2, 0))
    ax3.set_title('Negative (Different person)')
    ax3.imshow(N.permute(1, 2, 0))
    plt.suptitle('Triplet Sample: Anchor-Positive-Negative', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'triplet_sample.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create dataloaders with optimizations
    train_loader = DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        valset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    print(f"\n   Batches in train_loader: {len(train_loader)}")
    print(f"   Batches in val_loader: {len(val_loader)}")
    
    # Initialize model
    print("\n🧠 Initializing Siamese Network (EfficientNet-B0 backbone)...")
    model = APN_Model()
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.TripletMarginLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    
    # Training loop with early stopping
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
        
        # Update learning rate scheduler
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if valid_loss < best_valid_loss:
            model_path = os.path.join(MODEL_DIR, 'best_model.pt')
            # Save comprehensive checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'best_valid_loss': valid_loss,
            }, model_path)
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            print(f'   ✅ EPOCH {epoch+1}/{EPOCHS} | train: {train_loss:.4f} | valid: {valid_loss:.4f} | lr: {current_lr:.6f} | SAVED!')
        else:
            epochs_without_improvement += 1
            print(f'   ⏳ EPOCH {epoch+1}/{EPOCHS} | train: {train_loss:.4f} | valid: {valid_loss:.4f} | lr: {current_lr:.6f} | No improvement ({epochs_without_improvement}/{EARLY_STOP_PATIENCE})')
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs!")
            break
    
    # Plot training history
    print("\n📈 Saving training history plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(valid_losses, label='Valid Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate embeddings database
    print("\n📊 Generating embeddings database...")
    model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded best model from epoch {checkpoint['epoch']} (valid_loss: {checkpoint['valid_loss']:.4f})")
    
    df_enc = get_encoding_csv(model, df['Anchor'], DATA_DIR)
    
    csv_path = os.path.join(MODEL_DIR, 'database.csv')
    df_enc.to_csv(csv_path, index=False)
    print(f"   Embeddings saved to: {csv_path}")
    
    # Test re-identification
    print("\n🔍 Testing person re-identification...")
    idx = 0
    img_name = df_enc['Anchor'].iloc[idx]
    img_path = os.path.join(DATA_DIR, img_name)
    
    img = io.imread(img_path)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
    model.eval()
    with torch.no_grad():
        img = img.to(DEVICE)
        img_enc = model(img.unsqueeze(0))
        img_enc = img_enc.squeeze().cpu().detach().numpy()
        anc_enc_arr = df_enc.iloc[:, 1:].to_numpy()
        anc_img_names = df_enc['Anchor']
    
    # Compute distances
    distance = []
    for i in range(anc_enc_arr.shape[0]):
        dist = euclidean_dist(img_enc, anc_enc_arr[i:i+1, :])
        distance = np.append(distance, dist)
    
    closest_idx = np.argsort(distance)
    
    # Visualize results
    print("   Plotting closest matches...")
    plot_closest_imgs_simple(anc_img_names, DATA_DIR, img_path, closest_idx, distance, no_of_closest=5)
    
    print("\n✅ Done! Check the generated images:")
    print(f"   - triplet_sample.png")
    print(f"   - results.png")


if __name__ == "__main__":
    main()

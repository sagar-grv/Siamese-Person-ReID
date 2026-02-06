"""
Person Re-Identification Streamlit App
Compare Triplet Loss vs Contrastive Loss models
Upload an image to find matching persons from the database
"""

import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from PIL import Image
import os

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths (updated for new folder structure)
MODELS = {
    "Triplet Loss": {
        "model_path": os.path.join(SCRIPT_DIR, 'models', 'triplet', 'best_model.pt'),
        "database_path": os.path.join(SCRIPT_DIR, 'models', 'triplet', 'database.csv'),
        "description": "Uses anchor, positive, and negative samples. Learns to minimize distance between anchor-positive and maximize distance between anchor-negative.",
        "color": "#667eea"
    },
    "Contrastive Loss": {
        "model_path": os.path.join(SCRIPT_DIR, 'models', 'contrastive', 'best_model_contrastive.pt'),
        "database_path": os.path.join(SCRIPT_DIR, 'models', 'contrastive', 'database_contrastive.csv'),
        "description": "Uses positive and negative pairs. Learns to minimize distance for similar pairs and maximize distance for dissimilar pairs.",
        "color": "#f093fb"
    }
}

# Data directory (updated for new folder structure)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'SNN-TL-Data', 'train')

# ============================================================
# Model Definition (same for both losses)
# ============================================================
class SiameseModel(nn.Module):
    def __init__(self, emb_size=512):
        super(SiameseModel, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False)
        self.efficientnet.classifier = nn.Linear(
            in_features=self.efficientnet.classifier.in_features,
            out_features=emb_size
        )
    
    def forward(self, x):
        return self.efficientnet(x)

# ============================================================
# Helper Functions
# ============================================================
def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

@st.cache_resource
def load_model(model_path):
    """Load a trained model"""
    model = SiameseModel()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_data
def load_database(database_path):
    """Load embeddings database"""
    return pd.read_csv(database_path)

def process_image(image):
    """Process uploaded image for the model"""
    image = image.resize((64, 128))
    img_array = np.array(image).astype(np.float32) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    return img_tensor

def get_embedding(model, image_tensor):
    """Get embedding for an image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        embedding = model(image_tensor.unsqueeze(0))
        return embedding.squeeze().cpu().numpy()

def find_closest_matches(query_embedding, database_df, top_k=5):
    """Find closest matches in the database"""
    embeddings = database_df.iloc[:, 1:].to_numpy()
    image_names = database_df['Anchor'].tolist()
    
    distances = []
    for i in range(len(embeddings)):
        dist = euclidean_dist(query_embedding, embeddings[i])
        distances.append((i, dist, image_names[i]))
    
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]

# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(
        page_title="Person Re-ID: Loss Comparison",
        page_icon="🔍",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .loss-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 2px solid rgba(255,255,255,0.1);
    }
    .triplet-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .contrastive-badge {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .distance-badge {
        background: linear-gradient(135deg, #00c9ff, #92fe9d);
        color: #1a1a2e;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Person Re-Identification</h1>
        <p>Compare Triplet Loss vs Contrastive Loss</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Loss function selection
        st.subheader("📊 Loss Function")
        loss_type = st.radio(
            "Select model type:",
            list(MODELS.keys()),
            index=0
        )
        
        model_info = MODELS[loss_type]
        
        # Check if model exists
        model_exists = os.path.exists(model_info["model_path"])
        database_exists = os.path.exists(model_info["database_path"])
        
        if model_exists and database_exists:
            st.success(f"✅ {loss_type} model loaded")
        else:
            if not model_exists:
                st.error(f"❌ Model not found")
            if not database_exists:
                st.error(f"❌ Database not found")
        
        st.divider()
        
        # Description
        st.markdown(f"**About {loss_type}:**")
        st.caption(model_info["description"])
        
        st.divider()
        
        top_k = st.slider("Number of matches to show", 1, 10, 5)
        
        st.divider()
        st.markdown(f"**Device:** `{DEVICE}`")
        st.markdown(f"**Model:** EfficientNet-B0")
        
        # Compare mode
        st.divider()
        compare_mode = st.checkbox("🔄 Compare both models", value=False)
    
    # Check if selected model is available
    if not model_exists or not database_exists:
        st.error(f"❌ {loss_type} model is not available. Please train it first.")
        if loss_type == "Contrastive Loss":
            st.info("Run `python src/train_contrastive.py` to train the contrastive loss model.")
        else:
            st.info("Run `python src/train_triplet.py` to train the triplet loss model.")
        return
    
    # Load model and database
    with st.spinner(f"Loading {loss_type} model..."):
        model = load_model(model_info["model_path"])
        database_df = load_database(model_info["database_path"])
    
    # Also load comparison model if compare mode is on
    compare_model = None
    compare_database = None
    compare_loss_type = "Contrastive Loss" if loss_type == "Triplet Loss" else "Triplet Loss"
    
    if compare_mode:
        compare_info = MODELS[compare_loss_type]
        if os.path.exists(compare_info["model_path"]) and os.path.exists(compare_info["database_path"]):
            with st.spinner(f"Loading {compare_loss_type} model..."):
                compare_model = load_model(compare_info["model_path"])
                compare_database = load_database(compare_info["database_path"])
        else:
            st.warning(f"⚠️ {compare_loss_type} model not available for comparison")
            compare_mode = False
    
    st.success(f"✅ Loaded {len(database_df)} images in database")
    
    # File uploader - create columns based on mode
    if compare_mode and compare_model is not None:
        col1, col2, col3 = st.columns([1, 1, 1])
    else:
        col1, col2 = st.columns([1, 2])
        col3 = None
    
    with col1:
        st.subheader("📤 Upload Query Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a person image to find matches"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Query Image", width=200)
            
            if st.button("🔍 Find Matches", type="primary"):
                with st.spinner("Processing..."):
                    img_tensor = process_image(image)
                    
                    # Get matches from primary model
                    query_embedding = get_embedding(model, img_tensor)
                    matches = find_closest_matches(query_embedding, database_df, top_k)
                    
                    st.session_state['matches'] = matches
                    st.session_state['loss_type'] = loss_type
                    st.session_state['query_processed'] = True
                    
                    # Get matches from comparison model
                    if compare_mode and compare_model is not None:
                        compare_embedding = get_embedding(compare_model, img_tensor)
                        compare_matches = find_closest_matches(compare_embedding, compare_database, top_k)
                        st.session_state['compare_matches'] = compare_matches
    
    # Results columns
    if compare_mode and compare_model is not None and col3 is not None:
        with col2:
            st.subheader(f"🎯 {loss_type} Results")
            display_results(loss_type, top_k)
        
        with col3:
            st.subheader(f"🎯 {compare_loss_type} Results")
            display_compare_results(compare_loss_type, top_k)
    else:
        with col2:
            st.subheader(f"🎯 Top Matches ({loss_type})")
            display_results(loss_type, top_k)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🧠 Powered by Siamese Neural Network with EfficientNet-B0 backbone</p>
        <p><b>Triplet Loss:</b> anchor-positive-negative triplets | <b>Contrastive Loss:</b> positive-negative pairs</p>
    </div>
    """, unsafe_allow_html=True)


def display_results(loss_type, top_k):
    """Display match results"""
    if 'matches' in st.session_state and st.session_state.get('query_processed'):
        matches = st.session_state['matches']
        
        cols = st.columns(min(len(matches), 5))
        
        for i, (idx, dist, img_name) in enumerate(matches[:5]):
            with cols[i % 5]:
                img_path = os.path.join(DATA_DIR, img_name)
                if os.path.exists(img_path):
                    match_img = Image.open(img_path)
                    st.image(match_img, width=120)
                    badge_class = "triplet-badge" if "Triplet" in loss_type else "contrastive-badge"
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <span class='distance-badge'>Dist: {dist:.4f}</span>
                        <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #888;'>
                            #{i+1} Match
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        if len(matches) > 5:
            st.divider()
            remaining_cols = st.columns(min(len(matches) - 5, 5))
            for i, (idx, dist, img_name) in enumerate(matches[5:]):
                with remaining_cols[i % 5]:
                    img_path = os.path.join(DATA_DIR, img_name)
                    if os.path.exists(img_path):
                        match_img = Image.open(img_path)
                        st.image(match_img, width=120)
                        st.markdown(f"""
                        <div style='text-align: center;'>
                            <span class='distance-badge'>Dist: {dist:.4f}</span>
                            <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #888;'>
                                #{i+6} Match
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("👈 Upload an image and click 'Find Matches' to see results")


def display_compare_results(loss_type, top_k):
    """Display comparison model results"""
    if 'compare_matches' in st.session_state and st.session_state.get('query_processed'):
        matches = st.session_state['compare_matches']
        
        cols = st.columns(min(len(matches), 5))
        
        for i, (idx, dist, img_name) in enumerate(matches[:5]):
            with cols[i % 5]:
                img_path = os.path.join(DATA_DIR, img_name)
                if os.path.exists(img_path):
                    match_img = Image.open(img_path)
                    st.image(match_img, width=120)
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <span class='distance-badge'>Dist: {dist:.4f}</span>
                        <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #888;'>
                            #{i+1} Match
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Results will appear here")


if __name__ == "__main__":
    main()

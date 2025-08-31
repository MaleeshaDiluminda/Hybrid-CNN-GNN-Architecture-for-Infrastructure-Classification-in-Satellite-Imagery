from __future__ import annotations
import os, io, base64
from dataclasses import dataclass
from typing import Tuple
from typing import cast

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory, render_template
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from werkzeug.utils import secure_filename

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_size: int = 224
    freeze_backbone: bool = False
    ckpt_path: str = "checkpoints/hybrid_cnn_gnn_best.pt"
    upload_dir: str = "uploads"
    allowed_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    max_content_length_mb: int = 10

cfg = CFG()
# Ensure upload directory exists
os.makedirs(cfg.upload_dir, exist_ok=True)

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

t_eval = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

LABELS = {0: "Non-Infrastructure", 1: "Infrastructure"}

# -----------------------------
# Build adjacency matrix
# -----------------------------
def make_grid_adjacency(H: int, W: int, self_loops=True):
    N = H * W
    edges = []
    def node(r, c): return r * W + c
    for r in range(H):
        for c in range(W):
            u = node(r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < H and 0 <= c2 < W:
                    v = node(r2, c2)
                    edges.append((u, v))
    A = torch.zeros((N, N), dtype=torch.float32)
    for u, v in edges:
        A[u, v] = 1.0
    if self_loops:
        A += torch.eye(N)
    deg = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5).clamp(min=1e-6))
    return D_inv_sqrt @ A @ D_inv_sqrt

# -----------------------------
# Hybrid CNN-GNN Model
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activation=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, X, A_hat):
        X = self.lin(X)
        X = torch.einsum("ij,bjd->bid", A_hat, X)
        X = self.act(X)
        return self.dropout(X)

class CNN_GNN(nn.Module):
    def __init__(self, freeze_backbone=False, g_hidden=256, g_out=128, dropout=0.1):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feat_dim = 512

        self.g1 = GCNLayer(self.feat_dim, g_hidden, dropout=dropout, activation=True)
        self.g2 = GCNLayer(g_hidden, g_out, dropout=dropout, activation=True)
        self.head = nn.Linear(g_out, 1)

        A_hat = make_grid_adjacency(7, 7, self_loops=True)
        self.register_buffer("A_hat", A_hat)

    def forward(self, x):
        fmap = self.stem(x)
        fmap = self.pool(fmap)
        X = fmap.flatten(2).transpose(1, 2)  # (B, 49, 512)
        X = self.g1(X, self.A_hat)
        X = self.g2(X, self.A_hat)
        X = X.mean(dim=1)
        return self.head(X).squeeze(1)

class CNN_GNN_NoPreTrained(nn.Module):
    """Fallback model that doesn't require pre-trained weights download"""
    def __init__(self, freeze_backbone=False, g_hidden=256, g_out=128, dropout=0.1):
        super().__init__()
        backbone = resnet18(weights=None)  # No pre-trained weights
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feat_dim = 512

        self.g1 = GCNLayer(self.feat_dim, g_hidden, dropout=dropout, activation=True)
        self.g2 = GCNLayer(g_hidden, g_out, dropout=dropout, activation=True)
        self.head = nn.Linear(g_out, 1)

        A_hat = make_grid_adjacency(7, 7, self_loops=True)
        self.register_buffer("A_hat", A_hat)

    def forward(self, x):
        fmap = self.stem(x)
        fmap = self.pool(fmap)
        X = fmap.flatten(2).transpose(1, 2)  # (B, 49, 512)
        X = self.g1(X, self.A_hat)
        X = self.g2(X, self.A_hat)
        X = X.mean(dim=1)
        return self.head(X).squeeze(1)

# -----------------------------
# Load trained model once
# -----------------------------
def load_hybrid_model():
    print(f"Loading model from: {cfg.ckpt_path}")
    
    # Ensure cache directory exists and is writable
    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for PyTorch cache
    os.environ['TORCH_HOME'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    try:
        model = CNN_GNN(freeze_backbone=cfg.freeze_backbone).to(cfg.device)
    except Exception as e:
        print(f"Error creating model: {e}")
        # Fallback: try without pre-trained weights if download fails
        print("Attempting to create model without pre-trained weights...")
        try:
            # Create model with random weights if pre-trained download fails
            model = CNN_GNN_NoPreTrained(freeze_backbone=cfg.freeze_backbone).to(cfg.device)
        except Exception as e2:
            print(f"Fallback model creation also failed: {e2}")
            raise e  # Re-raise original error
    
    if not os.path.isfile(cfg.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
    print(f"Model checkpoint found, loading state dict...")
    state = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Model loaded successfully on device: {cfg.device}")
    return model

# Global model variable for caching
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        try:
            print("Starting model loading...")
            MODEL = load_hybrid_model()
            print("Model loading completed!")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    return MODEL

# -----------------------------
# Prediction helper
# -----------------------------
@torch.no_grad()
def predict_image_bytes(img_bytes: bytes, threshold=0.5):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = cast(torch.Tensor, t_eval(img))
    x = tensor.unsqueeze(0).to(cfg.device)
    model = get_model()  # Get model lazily
    prob = torch.sigmoid(model(x)).item()
    pred = 1 if prob >= threshold else 0
    
    # Create base64 encoded image for preview
    img_buffer = io.BytesIO()
    # Resize image for preview (max 800px width to keep reasonable size)
    img_width, img_height = img.size
    if img_width > 800:
        ratio = 800 / img_width
        new_width = 800
        new_height = int(img_height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img.save(img_buffer, format='JPEG', quality=85, optimize=True)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return {
        "label": LABELS[pred],
        "pred_class": int(pred),
        "p_infrastructure": float(prob),
        "threshold": float(threshold),
        "image_data": img_base64,
    }

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in cfg.allowed_exts

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length_mb * 1024 * 1024



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')

    f = request.files.get("file")
    if not f or not f.filename or f.filename == "":
        return render_template('index.html', error="No file provided.")
    if not allowed_file(f.filename):
        return render_template('index.html', error=f"Unsupported file type. Allowed: {', '.join(cfg.allowed_exts)}")

    try:
        # Process image directly from memory
        img_bytes = f.read()
        result = predict_image_bytes(img_bytes, threshold=0.5)
        return render_template('index.html', result=result)
        
    except Exception as e:
        return render_template('index.html', error=f"Processing error: {str(e)}")

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(cfg.upload_dir, filename, as_attachment=False)

# JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Example:
    curl -X POST http://127.0.0.1:5000/api/predict -F "file=@/path/to/image.jpg"
    """
    f = request.files.get("file")
    if not f or not f.filename or f.filename == "":
        return jsonify({"error": "No file provided."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {cfg.allowed_exts}"}), 400

    try:
        result = predict_image_bytes(f.read(), threshold=0.5)
        # Remove image_data from API response to keep it lightweight
        api_result = {k: v for k, v in result.items() if k != 'image_data'}
        return jsonify(api_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask application for Hugging Face Spaces...")
    print(f"Device being used: {cfg.device}")
    print(f"Upload directory: {cfg.upload_dir}")
    try:
        # Use environment variable for port (Hugging Face Spaces uses port 7860)
        port = int(os.environ.get('PORT', 7860))
        host = os.environ.get('HOST', '0.0.0.0')
        debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        print(f"Starting server on {host}:{port}")
        app.run(host=host, port=port, debug=debug_mode)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        import traceback
        traceback.print_exc()
        raise

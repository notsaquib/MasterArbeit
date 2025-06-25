import os
import sys
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.load_splits import load_split

# ---- SETUP ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIR = "action_prediction_baseline/cached/features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# --- Load TimeSformer ---
# You can use pretrained weights from HuggingFace or timm

from timesformer.models.vit import TimeSformer

timesformer = TimeSformer(
    img_size=224,
    num_classes=400,
    num_frames=8,
    attention_type='divided_space_time'
)
timesformer.to(device)
timesformer.eval()

# Optionally load pretrained weights
# weights = torch.load("timesformer_k400.pyth", map_location="cpu")["model_state"]
# timesformer.load_state_dict(weights, strict=False)

# --- Define preprocessing transform for video ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features_from_video(frames):
    """
    Given video frames (T, H, W, 3), return a clip-level feature from TimeSformer.
    """
    clip_len = 8  # TimeSformer default
    T = len(frames)
    if T < clip_len:
        # Pad by repeating last frame
        pad_frames = [frames[-1]] * (clip_len - T)
        frames = list(frames) + pad_frames
    elif T > clip_len:
        # Uniformly sample
        indices = np.linspace(0, T - 1, clip_len).astype(int)
        frames = [frames[i] for i in indices]

    # Transform each frame
    processed = [transform(frame) for frame in frames]  # (C, H, W)
    video_tensor = torch.stack(processed, dim=1).unsqueeze(0).to(device)  # (1, 3, T, H, W)

    with torch.no_grad():
        out = timesformer(video_tensor)  # Output shape: (1, 400) by default
        feat = out.squeeze(0).cpu().numpy()  # Shape: (400,)

    return feat

def process_split(split, datasets):
    print(f"\n🔄 Processing {split.upper()} split...")
    data = load_split(split, datasets)

    features, labels, encoded_labels, sources, videos = [], [], [], [], []

    for sample in tqdm(data):
        frames = sample["features"]  # (T, H, W, 3)
        feat = extract_features_from_video(frames)
        features.append(feat)
        labels.append(sample["label"])
        encoded_labels.append(sample["encoded_label"])
        sources.append(sample["source"])
        videos.append(sample["video"])

    tag = "_".join(sorted(datasets))
    save_path = os.path.join(FEATURE_DIR, f"{split}_{tag}_timesformer.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "features": features,
            "labels": labels,
            "encoded_labels": encoded_labels,
            "sources": sources,
            "videos": videos
        }, f)
    print(f"✅ Saved features to: {save_path} ({len(features)} samples)")

if __name__ == "__main__":
    DATASETS = ["CHARADES", "MA52", "MMAct", "NTU", "PKU"]
    for split in ["train", "val", "test"]:
        process_split(split, DATASETS)

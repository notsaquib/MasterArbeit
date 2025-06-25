import os
import sys
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.load_splits import load_split  # Make sure this exists and works

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output folder
FEATURE_DIR = "action_prediction_baseline/cached/features/resnet50"
os.makedirs(FEATURE_DIR, exist_ok=True)

# Define image transform (to match ResNet input)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Converts (H, W, 3) uint8 → (3, H, W) float32 in [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load pretrained model
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
resnet.to(device)
resnet.eval()

def extract_features_from_video(frames):
    """
    Given a list or array of frames (T, H, W, 3), return a feature vector.
    Uses mean pooling over frame-wise ResNet features.
    """
    frame_tensors = [transform(frame) for frame in frames]  # No unsqueeze
    batch = torch.stack(frame_tensors).to(device)  # Shape: (T, 3, 224, 224)

    with torch.no_grad():
        feats = resnet(batch)  # Output shape: (T, 2048, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)  # (T, 2048)
        video_feat = feats.mean(dim=0).cpu().numpy()  # (2048,)
    
    return video_feat


def process_split(split, datasets):
    print(f"\n🔄 Processing {split.upper()} split...")
    data = load_split(split, datasets)

    features = []
    labels = []
    encoded_labels = []
    sources = []
    videos = []

    for sample in tqdm(data):
        frames = sample["features"]  # (T, H, W, 3)
        feat = extract_features_from_video(frames)
        features.append(feat)
        labels.append(sample["label"])
        encoded_labels.append(sample["encoded_label"])
        sources.append(sample["source"])
        videos.append(sample["video"])

    # Save feature dict
    tag = "_".join(sorted(datasets))
    save_path = os.path.join(FEATURE_DIR, f"{split}_{tag}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "features": features,
            "labels": labels,
            "encoded_labels" : encoded_labels,
            "sources": sources,
            "videos": videos
        }, f)
    print(f"✅ Saved features to: {save_path} ({len(features)} samples)")

if __name__ == "__main__":
    # Datasets to use
    DATASETS = ["CHARADES","MA52","MMAct","NTU","PKU"]  # Update as needed

    for split in ["train", "val", "test"]:
        process_split(split, DATASETS)

import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from utils.mappings import ACTION_CLASS_MAPPING
import pickle

# Update with your actual MMAct dataset root
MMACT_ROOT = "Z:/MasterArbeit/Datasets/MMAct/Data_filtered"
CACHE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached")
CACHE_FILE = os.path.join(CACHE_FOLDER, "mmact_data.pkl")

# Target action labels (from filename, lowercase match)
MMACT_ACTIONS = list(ACTION_CLASS_MAPPING["MMAct"].keys())
DESIRED_ACTIONS = list(ACTION_CLASS_MAPPING["MMAct"].values())
print(f"MMACT_ACTIONS: {MMACT_ACTIONS}")


def extract_full_video(video_path, max_frames=30, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # Pad or sample
    frames = np.array(frames)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = frames[idx]
    else:
        pad_len = max_frames - len(frames)
        frames = np.pad(frames, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')

    return frames

def load_mmact_data(max_frames=30):
    data = []
    print("📂 Loading MMAct dataset...")

    for root, _, files in tqdm(os.walk(MMACT_ROOT), desc="🔍 Scanning folders"):
        for file in files:
            if not file.endswith(".mp4"):
                continue

            label = os.path.splitext(file)[0].lower()  # e.g., "looking_around"
            if label not in MMACT_ACTIONS:
                continue
            
            index = MMACT_ACTIONS.index(label)
            desired_label_name = DESIRED_ACTIONS[index]
            video_path = os.path.join(root, file)
            segment = extract_full_video(video_path, max_frames=max_frames)

            if segment is not None:
                data.append({
                    "features": segment.astype(np.uint8),
                    "label": desired_label_name,
                    "source": "MMAct",
                    "video": file
                })

    print(f"✅ MMAct Loaded: {len(data)} samples.")

    # Save to cache
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Saved cached data to {CACHE_FILE}")

    return data

if __name__ == "__main__":
    sample_data = load_mmact_data()
    print(f"🔢 Loaded {len(sample_data)} video samples.")

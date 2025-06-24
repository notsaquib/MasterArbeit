import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle

# Add root for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mappings import ACTION_CLASS_MAPPING

# Constants
MA52_ROOT = "Z:/MasterArbeit/Datasets/MA-52"
VIDEO_FOLDER = os.path.join(MA52_ROOT, "train")
ANNOTATION_FILE = os.path.join(MA52_ROOT, "annotations/train_list_videos.txt")
CACHE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached")
CACHE_FILE = os.path.join(CACHE_FOLDER, "ma52_data.pkl")

# All allowed action IDs
MA52_ACTION_IDS = list(map(int, ACTION_CLASS_MAPPING["MA-52"].keys()))
DESIRED_ACTIONS = list(ACTION_CLASS_MAPPING["MA-52"].values())
print(f"MA52_ACTION_IDS: {MA52_ACTION_IDS}")

def extract_full_video(video_path, max_frames=30, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()

    # Normalize to fixed number of frames
    if len(frames) == 0:
        return None

    frames = np.array(frames)

    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = frames[idx]
    else:
        pad_len = max_frames - len(frames)
        frames = np.pad(frames, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')

    return frames

def load_ma52_data(max_frames=30):
    print("📂 Loading MA-52 Dataset...")
    data = []

    with open(ANNOTATION_FILE, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="🔄 Processing videos"):
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        video_file, label_id = parts[0], int(parts[1])

        if label_id not in MA52_ACTION_IDS:
            continue

        index = MA52_ACTION_IDS.index(label_id)
        desired_label_name = DESIRED_ACTIONS[index]
        video_path = os.path.join(VIDEO_FOLDER, video_file)

        if not os.path.exists(video_path):
            print(f"⚠️ Skipping missing file: {video_file}")
            continue

        segment = extract_full_video(video_path, max_frames=max_frames)
        if segment is None:
            print(f"⚠️ Could not extract: {video_file}")
            continue

        data.append({
            "features": segment.astype(np.uint8),  # shape: (max_frames, H, W, 3)
            "label": desired_label_name,
            "source": "MA-52",
            "video": video_file
        })

    print(f"✅ MA-52 Loaded: {len(data)} samples.")

    # Save to cache
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Saved cached data to {CACHE_FILE}")

    return data

# For manual testing
if __name__ == "__main__":
    load_ma52_data()

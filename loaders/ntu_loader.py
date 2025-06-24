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
NTU_ROOT = "Z:/MasterArbeit/Datasets/NTU"
VIDEO_FOLDER = os.path.join(NTU_ROOT, "Data_filtered")  # adjust this if needed
CACHE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached")
CACHE_FILE = os.path.join(CACHE_FOLDER, "ntu_data.pkl")

# All allowed action IDs
NTU_ACTION_IDS = list(ACTION_CLASS_MAPPING["NTU"].keys())
DESIRED_ACTIONS = list(ACTION_CLASS_MAPPING["NTU"].values())
print(f"NTU_ACTION_IDS: {NTU_ACTION_IDS}")

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

def load_ntu_data(max_frames=30):
    print("📂 Loading NTU RGB Dataset...")
    data = []

    for filename in tqdm(os.listdir(VIDEO_FOLDER), desc="🔄 Processing videos"):
        if not filename.endswith("_rgb.avi"):
            continue

        # Extract action ID, e.g., "A009" from "S001C001P001R001A009_rgb.avi"
        try:
            action_id = filename.split("A")[1][:3]  # Get '009' from 'A009'
            action_token = f"A{action_id}"
        except Exception:
            print(f"⚠️ Skipping unrecognized filename format: {filename}")
            continue

        if action_token not in NTU_ACTION_IDS:
            continue

        index = NTU_ACTION_IDS.index(action_token)
        desired_label_name = DESIRED_ACTIONS[index]    
        video_path = os.path.join(VIDEO_FOLDER, filename)
        segment = extract_full_video(video_path, max_frames=max_frames)

        if segment is None:
            print(f"⚠️ Could not extract: {filename}")
            continue

        data.append({
            "features": segment.astype(np.uint8),
            "label": desired_label_name,
            "source": "NTU",
            "video": filename
        })

    print(f"✅ NTU RGB Loaded: {len(data)} samples.")

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Saved cached data to {CACHE_FILE}")

    return data

if __name__ == "__main__":
    load_ntu_data()

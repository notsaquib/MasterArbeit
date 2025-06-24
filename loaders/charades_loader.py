import os
import sys
import cv2
import csv
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.mappings import ACTION_CLASS_MAPPING


CACHE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached")
CACHE_FILE = os.path.join(CACHE_FOLDER, "charades_data.pkl")

# Charades paths (adjust these)
CHARADES_ROOT = "C:/Users/khan/Downloads/Charades"
VIDEO_DIR = os.path.join(CHARADES_ROOT, "raw_data")
ANNOTATION_FILE = os.path.join(CHARADES_ROOT, "annotations", "Charades_v1_train.csv")

# Desired class codes (e.g., c038 -> "standing up")
DESIRED_ACTIONS = ACTION_CLASS_MAPPING["Charades"]
DESIRED_CLASS_IDS = set(DESIRED_ACTIONS.keys())

def extract_segment(video_path, start_sec, end_sec, max_frames=30, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        return None

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    segment = []
    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        segment.append(frame)

    cap.release()
    if not segment:
        return None

    # Normalize to max_frames
    segment = np.array(segment)
    if len(segment) > max_frames:
        idx = np.linspace(0, len(segment) - 1, max_frames).astype(int)
        segment = segment[idx]
    else:
        pad_len = max_frames - len(segment)
        segment = np.pad(segment, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')

    return segment

def load_charades_rgb_data(max_frames=30):
    data = []
    with open(ANNOTATION_FILE, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Loading Charades"):
            video_id = row["id"]
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue

            actions_str = row["actions"]
            if not actions_str.strip():
                continue

            actions = actions_str.split(";")
            for act in actions:
                parts = act.strip().split()
                if len(parts) != 3:
                    continue
                cls, start, end = parts
                if cls not in DESIRED_CLASS_IDS:
                    continue

                start = float(start)
                end = float(end)
                segment = extract_segment(video_path, start, end, max_frames=max_frames)

                if segment is not None:
                    data.append({
                        "features": segment.astype(np.uint8),
                        "label": DESIRED_ACTIONS[cls],
                        "source": "Charades",
                        "video": video_id
                    })
                print(data)

    print(f"\n✅ Load Complete: {len(data)} segments.")
    
     # Save to cache
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Saved cached data to {CACHE_FILE}")

    return data

if __name__ == "__main__":
    load_charades_rgb_data()

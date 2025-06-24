
import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from utils.mappings import ACTION_CLASS_MAPPING
from tqdm import tqdm
import pickle

# Root PKU path (update as needed)
PKU_ROOT = "Z:/MasterArbeit/Datasets/PKU-MMD/Data"
CACHE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached")
CACHE_FILE = os.path.join(CACHE_FOLDER, "pku_data.pkl")

# Target action labels to extract
PKU_ACTION_IDS = list(ACTION_CLASS_MAPPING["PKU"].keys())
DESIRED_ACTIONS = list(ACTION_CLASS_MAPPING["PKU"].values())

def extract_video_segment(video_path, start_frame, end_frame, max_frames=30, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    → Extracting frames from: {video_path}, start={start_frame}, end={end_frame}, total_frames={total_frames}")
    segment = []

    end_frame = min(end_frame, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"    ⚠️ Frame read failed at {i}")
            break
        frame = cv2.resize(frame, resize)
        segment.append(frame)

    cap.release()

    segment = np.array(segment)
    if len(segment) == 0:
        print("    ⚠️ No frames extracted. Skipping this sample.")
        return None

    if len(segment) > max_frames:
        idx = np.linspace(0, len(segment) - 1, max_frames).astype(int)
        segment = segment[idx]
    else:
        pad_len = max_frames - len(segment)
        segment = np.pad(segment, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')

    return segment


def load_pku_data(max_frames=30):
    rgb_dir = os.path.join(PKU_ROOT, "RGB")
    label_dir = os.path.join(PKU_ROOT, "Labels")


    print(f"📂 Looking for label files in: {label_dir}")
    print(f"📽️ Looking for video files in: {rgb_dir}")
    data = []
    total_labels = 0
    matched_labels = 0
    extracted_segments = 0

    for fname in tqdm(os.listdir(label_dir), desc="🔄 Scanning label files"):
        if not fname.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, fname)
        video_name = fname.replace(".txt", ".avi")
        video_path = os.path.join(rgb_dir, video_name)

        if not os.path.exists(video_path):
            print(f"❌ Missing video: {video_name}, skipping...")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            total_labels += 1
            parts = line.strip().split(',')
            if len(parts) != 4:
                print(f"⚠️ Invalid line format in {fname}: {line.strip()}")
                continue

            label_id, start, end, _ = map(int, parts)
            if str(label_id) not in PKU_ACTION_IDS:
                print(f"Skipping label_id {label_id} - not in PKU_ACTION_IDS")
                continue

            index = PKU_ACTION_IDS.index(str(label_id))
            desired_label_name = DESIRED_ACTIONS[index]
            matched_labels += 1

            segment = extract_video_segment(video_path, start, end, max_frames=max_frames)
            if segment is not None:
                print(f"Successfully loaded segment for {video_name}, label: {desired_label_name}")
                data.append({
                    "features": segment.astype(np.uint8),
                    "label": desired_label_name,
                    "source": "PKU",
                    "video": video_name
                })
                extracted_segments += 1
            else:
                print(f"Failed to extract segment for {video_name}, label: {desired_label_name}")

    print(f"\n✅ Load Complete: {extracted_segments} segments from {matched_labels}/{total_labels} labels.")

    # Save to cache
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Saved cached data to {CACHE_FILE}")

    return data


if __name__ == "__main__":
    print("✅ Script started")
    data = load_pku_data(max_frames=30)
    print(f"🔢 Total segments loaded: {len(data)}")
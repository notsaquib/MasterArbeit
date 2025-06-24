import os
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Constants
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached")  # Go up one level to action_prediction_baseline
DATASETS = ["ma52", "pku", "mmact", "charades", "ntu"]  # Add more if needed
#DATASETS = ["ma52"]
TEST_SPLIT = 0.15
VAL_SPLIT = 0.1
SEED = 4 #previously 42


def load_cached_dataset(dataset_name):
    path = os.path.join(CACHE_DIR, f"{dataset_name}_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Cache not found for {dataset_name}: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def consolidate_all_data():
    print("📦 Consolidating datasets from cache...")

    all_data = []
    for ds in DATASETS:
        print(f"📁 Loading cached dataset: {ds}")
        dataset = load_cached_dataset(ds)
        all_data.extend(dataset)

    print(f"✅ Total samples loaded: {len(all_data)}")
    return all_data


def encode_labels(data):
    print("🔠 Encoding labels...")
    labels = [sample["label"] for sample in data]
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels)

    for sample, enc_label in zip(data, encoded):
        sample["encoded_label"] = enc_label

    print(f"✅ Labels encoded: {label_encoder.classes_}")
    return data, label_encoder


def split_data(data):
    print("✂️ Splitting data into train/val/test...")
    labels = [sample["encoded_label"] for sample in data]

    # First split into train+val and test
    trainval_data, test_data = train_test_split(
        data, test_size=TEST_SPLIT, random_state=SEED, stratify=labels
    )

    # Split trainval into train and val
    trainval_labels = [sample["encoded_label"] for sample in trainval_data]
    train_data, val_data = train_test_split(
        trainval_data, test_size=VAL_SPLIT, random_state=SEED, stratify=trainval_labels
    )

    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data

def save_split_data(train_data, val_data, test_data, used_datasets):

    dataset_tag = "_".join(sorted(used_datasets))  # e.g., MA-52_MMACT_PKU

    with open(f"action_prediction_baseline/cached/train/train_{dataset_tag}.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(f"action_prediction_baseline/cached/val/val_{dataset_tag}.pkl", "wb") as f:
        pickle.dump(val_data, f)
    with open(f"action_prediction_baseline/cached/test/test_{dataset_tag}.pkl", "wb") as f:
        pickle.dump(test_data, f)

    print(f"📦 Saved splits to cached/consolidated/ with tag: {dataset_tag}")

def get_consolidated_data():
    all_data = consolidate_all_data()
    all_data, label_encoder = encode_labels(all_data)
    train_data, val_data, test_data = split_data(all_data)

    save_split_data(train_data, val_data, test_data, DATASETS)
    
    return train_data, val_data, test_data, label_encoder


# For testing
if __name__ == "__main__":
    train, val, test, le = get_consolidated_data()
    print(f"🧪 Encoded class labels: {le.classes_}")

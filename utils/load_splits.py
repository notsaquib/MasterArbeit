import os
import pickle

def load_split(split="train", datasets=["CHARADES","MA52","MMACT" "PKU"]):
    assert split in ["train", "val", "test"], "Invalid split type"
    
    tag = "_".join(sorted(d.lower() for d in datasets))  # e.g., "ma-52_pku"
    print((tag))
    file_path = f"action_prediction_baseline/cached/{split}/{split}_{tag}.pkl"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Split file not found: {file_path}")
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"📂 Loaded {split} set with {len(data)} samples for: {tag}")
    return data



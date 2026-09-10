import os
import requests
import pandas as pd
from typing import Dict, Tuple

SEP28K_LABELS_URL = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_labels.csv"
SEP28K_EPISODES_URL = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_episodes.csv"

def fetch_sep28k_metadata(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Downloads official Apple SEP-28k label & episode metadata CSVs from:
    https://github.com/apple/ml-stuttering-events-dataset
    
    NO FALLBACK BRANCH. If download fails, prints reason, raises RuntimeError, and exits non-zero.
    """
    os.makedirs(data_dir, exist_ok=True)
    labels_path = os.path.join(data_dir, "SEP-28k_labels.csv")
    episodes_path = os.path.join(data_dir, "SEP-28k_episodes.csv")
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    if not os.path.exists(labels_path):
        print(f"[SEP-28k Data] Downloading official labels CSV from {SEP28K_LABELS_URL}...")
        try:
            resp = requests.get(SEP28K_LABELS_URL, headers=headers, timeout=30)
            if resp.status_code == 200:
                with open(labels_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
            else:
                raise RuntimeError(f"HTTP Error {resp.status_code} downloading {SEP28K_LABELS_URL}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL FAILURE downloading SEP-28k labels: {e}")

    if not os.path.exists(episodes_path):
        print(f"[SEP-28k Data] Downloading official episodes CSV from {SEP28K_EPISODES_URL}...")
        try:
            resp = requests.get(SEP28K_EPISODES_URL, headers=headers, timeout=30)
            if resp.status_code == 200:
                with open(episodes_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
            else:
                raise RuntimeError(f"HTTP Error {resp.status_code} downloading {SEP28K_EPISODES_URL}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL FAILURE downloading SEP-28k episodes: {e}")

    labels_df = pd.read_csv(labels_path)
    episodes_df = pd.read_csv(episodes_path, header=None)
    
    # Speaker identity is Episode (Show_EpId), Show is the clustering unit
    labels_df["speaker_id"] = labels_df.apply(lambda r: f"{r['Show']}_{r['EpId']}", axis=1)

    stats = {
        "total_nominal_clips": len(labels_df),
        "total_episodes": len(episodes_df),
        "unique_shows": len(labels_df["Show"].unique()),
        "unique_speakers": len(labels_df["speaker_id"].unique()),
        "labels_path": labels_path,
        "episodes_path": episodes_path
    }

    return labels_df, episodes_df, stats

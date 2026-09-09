import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Tuple

SEP28K_LABELS_URL = "https://raw.githubusercontent.com/apple/ml-sep28k/main/SEP-28k_labels.csv"

def fetch_sep28k_metadata(data_dir: str = "data") -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Downloads official Apple SEP-28k label metadata CSV.
    Reports actual obtained clip counts and label distribution breakdown.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "SEP-28k_labels.csv")
    
    retrieval_status = "LOCAL_CACHE"
    if not os.path.exists(csv_path):
        print(f"[SEP-28k Data] Fetching official labels CSV from {SEP28K_LABELS_URL}...")
        try:
            resp = requests.get(SEP28K_LABELS_URL, timeout=30)
            if resp.status_code == 200:
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                retrieval_status = "DOWNLOAD_SUCCESS"
            else:
                retrieval_status = f"DOWNLOAD_FAILED_HTTP_{resp.status_code}"
        except Exception as e:
            retrieval_status = f"DOWNLOAD_FAILED_ERROR_{str(e)}"

    if not os.path.exists(csv_path):
        # Fallback to local table if exists
        local_table = os.path.join(data_dir, "sep28_codec_metadata.csv")
        if os.path.exists(local_table):
            df = pd.read_csv(local_table)
            retrieval_status = "FALLBACK_LOCAL_TABLE"
        else:
            raise FileNotFoundError(f"SEP-28k labels file could not be downloaded and no local metadata exists at {csv_path}")
    else:
        df = pd.read_csv(csv_path)

    # Calculate label distributions
    total_clips = len(df)
    
    # Identify show / podcast column
    show_col = None
    for candidate in ["Show", "show", "show_id", "Podcast"]:
        if candidate in df.columns:
            show_col = candidate
            break
    if show_col is None:
        df["Show"] = df["clip_id"].apply(lambda x: str(x).split("_")[0] if "_" in str(x) else "UnknownShow")
        show_col = "Show"

    # Identify speaker / episode column
    speaker_col = None
    for candidate in ["EpId", "Episode", "speaker_id", "clip_id"]:
        if candidate in df.columns:
            speaker_col = candidate
            break
    if speaker_col is None:
        speaker_col = "clip_id"

    # Determine disfluency label distribution
    label_cols = [c for c in ["Fluent", "Block", "Prolongation", "SoundRep", "WordRep", "Interjection", "label"] if c in df.columns]
    
    stats = {
        "total_obtained_clips": total_clips,
        "unique_shows": df[show_col].nunique(),
        "unique_speakers_or_episodes": df[speaker_col].nunique(),
        "retrieval_status": retrieval_status,
        "columns": list(df.columns)
    }

    return df, stats

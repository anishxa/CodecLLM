import os
import json
import pandas as pd
from typing import Dict, List, Any

def generate_5_fold_loso_splits(retrieved_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generates Leave-One-Show-Out cross-validation folds with Train, Dev, and Test splits.
    - Test show: fold index show
    - Dev show: next show in cyclic order (held out for threshold selection)
    - Train shows: remaining shows
    - Speaker identity: episode ID (Show_EpId)
    - Verifies 0 overlap across Train / Dev / Test sets for both Shows and Speakers.
    """
    assert "Show" in retrieved_df.columns, "Show column missing in dataframe"
    if "speaker_id" not in retrieved_df.columns:
        retrieved_df["speaker_id"] = retrieved_df["Show"].astype(str) + "_" + retrieved_df["EpId"].astype(str)
        
    surviving_shows = sorted(retrieved_df["Show"].unique().tolist())
    num_shows = len(surviving_shows)
    assert num_shows >= 3, f"Need at least 3 surviving shows for train/dev/test splits, found {num_shows}"
    
    folds = []
    for fold_idx in range(num_shows):
        test_show = surviving_shows[fold_idx]
        dev_show = surviving_shows[(fold_idx + 1) % num_shows]
        train_shows = [s for s in surviving_shows if s != test_show and s != dev_show]
        
        test_df = retrieved_df[retrieved_df["Show"] == test_show].copy()
        dev_df = retrieved_df[retrieved_df["Show"] == dev_show].copy()
        train_df = retrieved_df[retrieved_df["Show"].isin(train_shows)].copy()
        
        train_shows_set = set(train_df["Show"].unique())
        dev_shows_set = set(dev_df["Show"].unique())
        test_shows_set = set(test_df["Show"].unique())
        
        train_spks = set(train_df["speaker_id"].unique())
        dev_spks = set(dev_df["speaker_id"].unique())
        test_spks = set(test_df["speaker_id"].unique())
        
        # Disjointness checks
        overlap_shows = (train_shows_set & test_shows_set) | (train_shows_set & dev_shows_set) | (dev_shows_set & test_shows_set)
        overlap_spks = (train_spks & test_spks) | (train_spks & dev_spks) | (dev_spks & test_spks)
        
        assert len(overlap_shows) == 0, f"Fold {fold_idx} show overlap detected: {overlap_shows}"
        assert len(overlap_spks) == 0, f"Fold {fold_idx} speaker overlap detected: {overlap_spks}"
        
        notes = ""
        if len(test_df) < 100:
            notes = f"Small test fold (N={len(test_df)} clips); reported with explicit N."
            
        fold_dict = {
            "fold": fold_idx,
            "test_show": test_show,
            "dev_show": dev_show,
            "train_shows": sorted(list(train_shows_set)),
            "num_train_clips": len(train_df),
            "num_dev_clips": len(dev_df),
            "num_test_clips": len(test_df),
            "num_train_shows": len(train_shows_set),
            "num_dev_shows": len(dev_shows_set),
            "num_test_shows": len(test_shows_set),
            "num_train_speakers": len(train_spks),
            "num_dev_speakers": len(dev_spks),
            "num_test_speakers": len(test_spks),
            "disjoint_verification_passed": True,
            "notes": notes
        }
        folds.append(fold_dict)
        
    return folds

def save_folds_manifest(folds: List[Dict[str, Any]], surviving_shows: List[str], manifest_dir: str = "manifest") -> str:
    """
    Saves the fold definitions to manifest/folds.json so Phase 2+ reads fixed folds.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    out_path = os.path.join(manifest_dir, "folds.json")
    
    payload = {
        "dataset": "SEP-28k",
        "num_folds": len(folds),
        "surviving_shows": surviving_shows,
        "folds": folds
    }
    
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        
    return out_path

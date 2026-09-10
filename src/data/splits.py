import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

OFFICIAL_SEP28K_SHOWS = [
    "WomenWhoStutter",
    "StutterTalk",
    "StutteringIsCool",
    "HeStutters",
    "MyStutteringLife",
    "StrongVoices",
    "IStutterSoWhat",
    "HVSA"
]

def create_leave_one_show_out_splits(
    df: pd.DataFrame,
    test_show: str = "WomenWhoStutter",
    dev_show: str = "StutterTalk",
    show_col: str = "Show",
    speaker_col: str = "speaker_id"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Leave-One-Show-Out split discipline for SEP-28k:
    - Clusters by Show (8 official podcast shows).
    - Speaker identity is Episode (Show_EpId).
    - Asserts 0 overlap across Train / Dev / Test sets.
    """
    assert show_col in df.columns, f"Show column {show_col} missing in dataframe"
    assert test_show in df[show_col].values, f"Test show {test_show} not found in dataframe"
    assert dev_show in df[show_col].values, f"Dev show {dev_show} not found in dataframe"
    
    test_df = df[df[show_col] == test_show].copy()
    dev_df = df[df[show_col] == dev_show].copy()
    train_df = df[(df[show_col] != test_show) & (df[show_col] != dev_show)].copy()
    
    # Verify 0 overlap
    train_shows = set(train_df[show_col].unique())
    dev_shows = set(dev_df[show_col].unique())
    test_shows = set(test_df[show_col].unique())
    
    train_spks = set(train_df[speaker_col].unique())
    dev_spks = set(dev_df[speaker_col].unique())
    test_spks = set(test_df[speaker_col].unique())
    
    overlap_shows = train_shows.intersection(test_shows).union(train_shows.intersection(dev_shows)).union(dev_shows.intersection(test_shows))
    overlap_spks = train_spks.intersection(test_spks).union(train_spks.intersection(dev_spks)).union(dev_spks.intersection(test_spks))
    
    assert len(overlap_shows) == 0, f"Show overlap detected: {overlap_shows}"
    assert len(overlap_spks) == 0, f"Speaker overlap detected: {overlap_spks}"
    
    split_info = {
        "split_method": "leave_one_show_out",
        "test_show": test_show,
        "dev_show": dev_show,
        "num_train_clips": len(train_df),
        "num_dev_clips": len(dev_df),
        "num_test_clips": len(test_df),
        "num_train_shows": len(train_shows),
        "num_dev_shows": len(dev_shows),
        "num_test_shows": len(test_shows),
        "num_train_speakers": len(train_spks),
        "num_dev_speakers": len(dev_spks),
        "num_test_speakers": len(test_spks),
        "disjoint_verification_passed": True
    }
    
    return train_df, dev_df, test_df, split_info

import os
import sys
import json
import time
import subprocess
import pandas as pd
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import RunLogger
from src.data.sep28k import fetch_sep28k_metadata

SURVIVING_SHOWS = ["HeStutters", "MyStutteringLife", "StutterTalk", "HVSA", "WomenWhoStutter"]

def get_present_clips_on_disk(clips_dir: str = "clips") -> set:
    """
    Builds the set of clip tuples (Show, EpId, ClipId) actually present as .wav files on disk.
    """
    present = set()
    if not os.path.exists(clips_dir):
        return present
        
    for root, dirs, files in os.walk(clips_dir):
        for file in files:
            if file.endswith(".wav"):
                rel = os.path.relpath(os.path.join(root, file), clips_dir)
                parts = rel.split(os.sep)
                if len(parts) == 3:
                    show = parts[0]
                    try:
                        epid = int(parts[1])
                        clip_id = int(os.path.splitext(parts[2])[0])
                        present.add((show, epid, clip_id))
                    except ValueError:
                        pass
    return present

def generate_5_fold_loso_splits(retrieved_df: pd.DataFrame) -> List[Dict[str, any]]:
    """
    Generates 5 Leave-One-Show-Out folds across the 5 surviving shows.
    Leaves test_show out as test fold, uses remaining 4 shows as train fold.
    Includes explicit N stats for WomenWhoStutter (N=80).
    Verifies 0 overlap between Train and Test sets.
    """
    folds = []
    
    for fold_idx, test_show in enumerate(SURVIVING_SHOWS):
        test_df = retrieved_df[retrieved_df["Show"] == test_show].copy()
        train_df = retrieved_df[retrieved_df["Show"] != test_show].copy()
        
        train_shows = set(train_df["Show"].unique())
        test_shows = set(test_df["Show"].unique())
        
        train_spks = set(train_df["speaker_id"].unique())
        test_spks = set(test_df["speaker_id"].unique())
        
        overlap_shows = train_shows.intersection(test_shows)
        overlap_spks = train_spks.intersection(test_spks)
        
        assert len(overlap_shows) == 0, f"Show overlap in Fold {fold_idx}: {overlap_shows}"
        assert len(overlap_spks) == 0, f"Speaker overlap in Fold {fold_idx}: {overlap_spks}"
        
        notes = ""
        if test_show == "WomenWhoStutter":
            notes = "Small test fold (N=80 clips); reported with explicit N."

        folds.append({
            "fold": fold_idx,
            "test_show": test_show,
            "num_train_clips": len(train_df),
            "num_test_clips": len(test_df),
            "num_train_shows": len(train_shows),
            "num_test_shows": len(test_shows),
            "num_train_speakers": len(train_spks),
            "num_test_speakers": len(test_spks),
            "disjoint_verification_passed": True,
            "notes": notes
        })
        
    return folds

def main():
    print("=" * 80)
    print("      WORK ORDER 0 — PHASE 0: CORRECTED RETRIEVAL & 5-FOLD LOSO SPLITS")
    print("=" * 80)
    
    config = {
        "experiment_name": "phase0_work_order_0",
        "seed": 42,
        "labels_url": "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_labels.csv",
        "episodes_url": "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_episodes.csv",
        "surviving_shows": SURVIVING_SHOWS
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"phase0_{timestamp}"
    logger = RunLogger(run_id=run_id, config=config)
    
    # 1. Fetch official metadata CSVs
    print("[1/4] Loading official labels & episodes CSVs...")
    labels_df, episodes_df, stats = fetch_sep28k_metadata(data_dir="data")
    
    logger.register_dataset_checksum("sep28k_labels", stats["labels_path"])
    logger.register_dataset_checksum("sep28k_episodes", stats["episodes_path"])

    # 2. Run retry pass acquisition script
    print("\n[2/4] Running retry pass audio acquisition script (fetch_sep28k.py)...")
    fetch_script = os.path.join("scripts", "fetch_sep28k.py")
    res = subprocess.run([sys.executable, fetch_script], check=False)
    if res.returncode != 0:
        raise RuntimeError(f"fetch_sep28k.py failed with exit code {res.returncode}")

    # 3. Filter labels_df to ONLY clips actually present on disk under clips/
    print("\n[3/4] Filtering labels dataframe to clips ACTUALLY PRESENT on disk under clips/...")
    present_clips = get_present_clips_on_disk(clips_dir="clips")
    
    labels_df["is_present"] = labels_df.apply(lambda r: (r["Show"], int(r["EpId"]), int(r["ClipId"])) in present_clips, axis=1)
    retrieved_df = labels_df[labels_df["is_present"]].copy()
    
    print(f"  -> Nominal Labels in Dataset: {len(labels_df):,}")
    print(f"  -> Clips Actually Retrieved & Present on Disk: {len(retrieved_df):,}")

    # 4. Emit 5-Fold Leave-One-Show-Out splits over the retrieved clips
    print("\n[4/4] Generating 5-Fold Leave-One-Show-Out splits across the 5 surviving shows...")
    folds_info = generate_5_fold_loso_splits(retrieved_df)
    
    for fold in folds_info:
        print(f"  * Fold {fold['fold']} [Test: {fold['test_show']:16s}]: Train Clips = {fold['num_train_clips']:5,d} ({fold['num_train_shows']} shows) | Test Clips = {fold['num_test_clips']:5,d} ({fold['num_test_shows']} show)")
        if fold['notes']:
            print(f"      Note: {fold['notes']}")

    metrics_summary = {
        "nominal_labels": len(labels_df),
        "actual_retrieved_clips_on_disk": len(retrieved_df),
        "surviving_shows_count": len(SURVIVING_SHOWS),
        "loso_5_folds": folds_info
    }
    logger.log_metrics(metrics_summary)
    artifact_json_path = logger.finalize(status="COMPLETED")
    
    print("\n" + "=" * 80)
    print(f"[Phase 0 Complete] Saved run artifact to {artifact_json_path}")
    print("GATE 0 — STOP HERE. Phase 0 acquisition & LOSO split generation complete.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

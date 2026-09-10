import os
import sys
import json
import time
import subprocess
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import RunLogger
from src.data.sep28k import fetch_sep28k_metadata
from src.data.splits import create_leave_one_show_out_splits

def main():
    print("=" * 80)
    print("      PHASE 0: REAL DATA ACQUISITION & RETRIEVAL VERIFICATION")
    print("=" * 80)
    
    config = {
        "experiment_name": "phase0_real_data_acquisition",
        "seed": 42,
        "labels_url": "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_labels.csv",
        "episodes_url": "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_episodes.csv",
        "split_method": "leave_one_show_out",
        "test_show": "WomenWhoStutter",
        "dev_show": "StutterTalk"
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"phase0_{timestamp}"
    logger = RunLogger(run_id=run_id, config=config)
    
    # 1. Fetch metadata CSVs from official Apple repository
    print("[1/3] Fetching official labels & episodes CSVs from apple/ml-stuttering-events-dataset...")
    labels_df, episodes_df, stats = fetch_sep28k_metadata(data_dir="data")
    
    logger.register_dataset_checksum("sep28k_labels", stats["labels_path"])
    logger.register_dataset_checksum("sep28k_episodes", stats["episodes_path"])
    
    print(f"  -> Total Nominal Clips in Dataset: {stats['total_nominal_clips']:,}")
    print(f"  -> Total Podcast Episodes: {stats['total_episodes']}")
    print(f"  -> Unique Shows: {stats['unique_shows']}")
    print(f"  -> Unique Speakers (Episodes): {stats['unique_speakers']}")

    # 2. Run real audio retrieval script (fetch_sep28k.py)
    print("\n[2/3] Executing parallel audio episode retrieval and 16kHz mono clip extraction...")
    fetch_script = os.path.join("scripts", "fetch_sep28k.py")
    res = subprocess.run([sys.executable, fetch_script], check=False)
    if res.returncode != 0:
        raise RuntimeError(f"fetch_sep28k.py failed with exit code {res.returncode}")

    # 3. Apply Leave-One-Show-Out Split Discipline
    print("\n[3/3] Generating Leave-One-Show-Out splits and verifying disjointness...")
    train_df, dev_df, test_df, split_info = create_leave_one_show_out_splits(
        labels_df,
        test_show=config["test_show"],
        dev_show=config["dev_show"]
    )
    
    print(f"  -> Train Split: {split_info['num_train_clips']:,} clips ({split_info['num_train_shows']} shows, {split_info['num_train_speakers']} speakers)")
    print(f"  -> Dev Split:   {split_info['num_dev_clips']:,} clips ({split_info['num_dev_shows']} show: {config['dev_show']}, {split_info['num_dev_speakers']} speakers)")
    print(f"  -> Test Split:  {split_info['num_test_clips']:,} clips ({split_info['num_test_shows']} show: {config['test_show']}, {split_info['num_test_speakers']} speakers)")
    print(f"  -> Disjointness Verification: {'PASSED (0 overlap)' if split_info['disjoint_verification_passed'] else 'FAILED'}")

    metrics_summary = {
        "nominal_clips": stats["total_nominal_clips"],
        "total_episodes": stats["total_episodes"],
        "splits": split_info
    }
    logger.log_metrics(metrics_summary)
    artifact_json_path = logger.finalize(status="COMPLETED")
    
    print(f"\n[Phase 0 Complete] Saved run artifact to {artifact_json_path}")
    print("=" * 80)
    print("  STOP HERE. Phase 0 acquisition complete. Please inspect manifest/phase0_report.md and play clips in clips/.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

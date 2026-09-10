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
from src.data.splits import generate_5_fold_loso_splits, save_folds_manifest

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

def generate_phase0_markdown_report(episodes_manifest: List[Dict], labels_df: pd.DataFrame, clips_dir: str = "clips", manifest_dir: str = "manifest") -> str:
    """
    Generates manifest/phase0_report.md as the VERY LAST step of Phase 0.
    Reads real on-disk clip counts and finalized manifest status entries.
    """
    total_episodes = len(episodes_manifest)
    successful_episodes = sum(1 for ep in episodes_manifest if ep["status"] == "SUCCESS")
    retrieval_rate_pct = (successful_episodes / total_episodes) * 100.0
    
    actual_wav_count = 0
    total_clips_bytes = 0
    for root, dirs, files in os.walk(clips_dir):
        for file in files:
            if file.endswith(".wav"):
                actual_wav_count += 1
                total_clips_bytes += os.path.getsize(os.path.join(root, file))
                
    total_size_mb = total_clips_bytes / (1024 * 1024)
    
    status_counts = {}
    for ep in episodes_manifest:
        st = ep["status"]
        if "QUARANTINED_TOO_SHORT" in st:
            cat = "QUARANTINED_TOO_SHORT"
        elif st.startswith("FAILED_"):
            cat = st
        else:
            cat = st
        status_counts[cat] = status_counts.get(cat, 0) + 1

    report_path = os.path.join(manifest_dir, "phase0_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase 0 Real Data Acquisition Report: SEP-28k\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  \n")
        f.write(f"**Source Repository:** `https://github.com/apple/ml-stuttering-events-dataset`  \n\n")
        f.write("---  \n\n")
        f.write("## 1. Episode Retrieval & Status Breakdown\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Total Podcast Episodes Attempted** | {total_episodes} |\n")
        f.write(f"| **Episodes Successfully Verified & Active** | {successful_episodes} |\n")
        f.write(f"| **Episode Retrieval Success Rate** | **{retrieval_rate_pct:.2f}%** |\n")
        f.write(f"| **Total Real 3-Second .wav Clips Extracted** | **{actual_wav_count:,}** |\n")
        f.write(f"| **Total Audio Disk Footprint (`clips/`)** | **{total_size_mb:.2f} MB** |\n\n")
        f.write("### HTTP & Quarantine Status Breakdown\n\n")
        f.write("| Status Category | Count |\n")
        f.write("| :--- | :--- |\n")
        for st_name, st_cnt in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| `{st_name}` | {st_cnt} |\n")

        f.write("\n---  \n\n")
        f.write("## 2. Surviving Shows Retrieval Table\n\n")
        f.write("| Show Name | Nominal Labels | Extracted Clips on Disk (`clips/`) | Retrieval Status |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        shows = sorted(labels_df["Show"].unique())
        for show in shows:
            nom_count = len(labels_df[labels_df["Show"] == show])
            extracted_show_count = 0
            show_clips_dir = os.path.join(clips_dir, show)
            if os.path.exists(show_clips_dir):
                for root, dirs, files in os.walk(show_clips_dir):
                    extracted_show_count += len([f for f in files if f.endswith(".wav")])
            f.write(f"| **{show}** | {nom_count:,} | {extracted_show_count:,} | {'Active' if extracted_show_count > 0 else 'Dead URLs'} |\n")

    return report_path

def main():
    print("=" * 80)
    print("      WORK ORDER 0 — PHASE 0: CORRECTED RETRIEVAL & 5-FOLD LOSO SPLITS")
    print("=" * 80)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"phase0_{timestamp}"
    
    # 1. Fetch official metadata CSVs
    print("[1/4] Loading official labels & episodes CSVs...")
    labels_df, episodes_df, stats = fetch_sep28k_metadata(data_dir="data")

    # 2. Run retry pass acquisition & audit scripts
    print("\n[2/4] Running audio acquisition retry pass (fetch_sep28k.py)...")
    fetch_script = os.path.join("scripts", "fetch_sep28k.py")
    subprocess.run([sys.executable, fetch_script], check=False)

    print("\n      Running clip audit & marginal recovery script (audit_and_clean_clips.py)...")
    clean_script = os.path.join("scripts", "audit_and_clean_clips.py")
    res = subprocess.run([sys.executable, clean_script], check=False)
    if res.returncode != 0:
        raise RuntimeError(f"audit_and_clean_clips.py failed with exit code {res.returncode}")

    # 3. Filter labels_df to ONLY clips actually present on disk under clips/
    print("\n[3/4] Filtering labels dataframe to clips ACTUALLY PRESENT on disk under clips/...")
    present_clips = get_present_clips_on_disk(clips_dir="clips")
    
    labels_df["is_present"] = labels_df.apply(lambda r: (r["Show"], int(r["EpId"]), int(r["ClipId"])) in present_clips, axis=1)
    retrieved_df = labels_df[labels_df["is_present"]].copy()
    retrieved_df["speaker_id"] = retrieved_df["Show"].astype(str) + "_" + retrieved_df["EpId"].astype(str)
    
    surviving_shows = sorted(retrieved_df["Show"].unique().tolist())
    
    print(f"  -> Nominal Labels in Dataset: {len(labels_df):,}")
    print(f"  -> Clips Actually Retrieved & Present on Disk: {len(retrieved_df):,}")
    print(f"  -> Surviving Shows ({len(surviving_shows)}): {surviving_shows}")

    # 4. Emit 5-Fold Leave-One-Show-Out splits (Train / Dev / Test)
    print("\n[4/4] Generating 5-Fold Leave-One-Show-Out splits (Train/Dev/Test) across surviving shows...")
    folds_info = generate_5_fold_loso_splits(retrieved_df)
    folds_manifest_path = save_folds_manifest(folds_info, surviving_shows, manifest_dir="manifest")
    print(f"  -> Saved fold definitions to {folds_manifest_path}")

    # Regenerate manifest/phase0_report.md as the VERY LAST step
    episodes_jsonl_path = os.path.join("manifest", "episodes.jsonl")
    episodes_manifest = []
    with open(episodes_jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                episodes_manifest.append(json.loads(line.strip()))
                
    report_md_path = generate_phase0_markdown_report(episodes_manifest, labels_df, clips_dir="clips", manifest_dir="manifest")
    print(f"  -> Regenerated {report_md_path} as the final step of Phase 0.")

    config = {
        "experiment_name": "phase0_work_order_0",
        "seed": 42,
        "labels_url": "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_labels.csv",
        "episodes_url": "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_episodes.csv",
        "surviving_shows": surviving_shows,
        "folds_manifest": folds_manifest_path,
        "phase0_report": report_md_path
    }
    
    logger = RunLogger(run_id=run_id, config=config)
    logger.register_dataset_checksum("sep28k_labels", stats["labels_path"])
    logger.register_dataset_checksum("sep28k_episodes", stats["episodes_path"])
    
    for fold in folds_info:
        print(f"  * Fold {fold['fold']} [Test: {fold['test_show']:16s} | Dev: {fold['dev_show']:16s}]: Train={fold['num_train_clips']:5,d} ({fold['num_train_shows']} shows) | Dev={fold['num_dev_clips']:5,d} | Test={fold['num_test_clips']:5,d}")
        if fold['notes']:
            print(f"      Note: {fold['notes']}")

    metrics_summary = {
        "nominal_labels": len(labels_df),
        "actual_retrieved_clips_on_disk": len(retrieved_df),
        "surviving_shows_count": len(surviving_shows),
        "surviving_shows": surviving_shows,
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

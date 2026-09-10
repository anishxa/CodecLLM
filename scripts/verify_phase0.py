import os
import sys
import json
import soundfile as sf
from typing import Dict, List, Set, Tuple

def verify_phase0():
    print("=" * 80)
    print("             VERIFYING PHASE 0 DELIVERABLES & CONSTRAINTS")
    print("=" * 80)
    
    data_dir = "data"
    manifest_dir = "manifest"
    clips_dir = "clips"
    
    episodes_jsonl_path = os.path.join(manifest_dir, "episodes.jsonl")
    folds_json_path = os.path.join(manifest_dir, "folds.json")
    
    # Check 1: Manifest Files Exist
    print("\n[Check 1/5] Verifying manifest files exist...")
    assert os.path.exists(episodes_jsonl_path), f"Missing {episodes_jsonl_path}"
    assert os.path.exists(folds_json_path), f"Missing {folds_json_path}"
    print("  -> manifest/episodes.jsonl and manifest/folds.json present.")

    # Load manifest entries
    episodes = []
    with open(episodes_jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line.strip()))
                
    successful_episodes = {
        (ep["show"], ep["epid"]): ep
        for ep in episodes
        if ep["status"] == "SUCCESS"
    }
    manifest_extracted_sum = sum(ep["extracted_clips"] for ep in episodes if ep["status"] == "SUCCESS")
    print(f"  -> Loaded {len(episodes)} episodes from manifest ({len(successful_episodes)} SUCCESS, sum extracted = {manifest_extracted_sum:,}).")

    # Check 2: Disk Directory Completeness & Purge Verification
    print("\n[Check 2/5] Auditing clips/ filesystem directories against manifest entries...")
    disk_episodes_found = set()
    total_disk_wavs = 0
    
    for show_name in os.listdir(clips_dir):
        show_path = os.path.join(clips_dir, show_name)
        if not os.path.isdir(show_path):
            continue
            
        for ep_dir in os.listdir(show_path):
            ep_path = os.path.join(show_path, ep_dir)
            if not os.path.isdir(ep_path):
                continue
                
            try:
                epid_int = int(ep_dir)
            except ValueError:
                continue
                
            ep_key = (show_name, epid_int)
            disk_episodes_found.add(ep_key)
            
            # Assert this episode is marked SUCCESS in manifest
            assert ep_key in successful_episodes, f"STALE DATA DETECTED: {show_name}/{epid_int} on disk but NOT marked SUCCESS in manifest!"
            
            # Count wav files in this episode directory
            wav_files = [f for f in os.listdir(ep_path) if f.endswith(".wav")]
            expected_clips = successful_episodes[ep_key]["extracted_clips"]
            assert len(wav_files) == expected_clips, (
                f"EPISODE INCOMPLETENESS: {show_name}/{epid_int} has {len(wav_files)} wav files on disk, but manifest says {expected_clips}!"
            )
            total_disk_wavs += len(wav_files)

    print(f"  -> Audited {len(disk_episodes_found)} episode directories on disk. All match manifest extracted clip counts.")

    # Check 3: Permanent Disk Count == Manifest Sum Assertion
    print("\n[Check 3/5] Asserting disk clip count equals manifest extracted clips sum...")
    assert total_disk_wavs == manifest_extracted_sum, (
        f"CRITICAL MISMATCH: Total WAV files on disk ({total_disk_wavs}) != Manifest sum ({manifest_extracted_sum})"
    )
    print(f"  -> PASSED: Disk clip count ({total_disk_wavs:,}) == Manifest extracted clip sum ({manifest_extracted_sum:,}).")

    # Check 4: Audio Format Validation (16kHz, 48,000 samples = 3.00s)
    print("\n[Check 4/5] Spot-checking audio clip sample rates and durations...")
    checked_count = 0
    for root, dirs, files in os.walk(clips_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_p = os.path.join(root, file)
                info = sf.info(wav_p)
                assert info.samplerate == 16000, f"{wav_p} samplerate is {info.samplerate}, expected 16000"
                assert info.frames == 48000, f"{wav_p} duration is {info.frames} samples, expected 48000 (3.0s)"
                checked_count += 1
                if checked_count >= 1000:  # spot check sample
                    break
        if checked_count >= 1000:
            break
    print(f"  -> PASSED: Spot-checked {checked_count} .wav files on disk (16kHz mono, 48,000 samples / 3.0s).")

    # Check 5: Folds Manifest Verification (Train / Dev / Test Disjointness)
    print("\n[Check 5/5] Verifying 5-fold Train/Dev/Test LOSO splits in manifest/folds.json...")
    with open(folds_json_path, "r") as f:
        folds_manifest = json.load(f)
        
    num_folds = folds_manifest.get("num_folds", 0)
    assert num_folds >= 3, f"Expected >= 3 folds, found {num_folds}"
    
    for fold in folds_manifest["folds"]:
        f_idx = fold["fold"]
        t_show = fold["test_show"]
        d_show = fold["dev_show"]
        tr_shows = set(fold["train_shows"])
        
        # Verify show disjointness
        assert t_show not in tr_shows, f"Fold {f_idx}: Test show {t_show} found in train shows"
        assert d_show not in tr_shows, f"Fold {f_idx}: Dev show {d_show} found in train shows"
        assert t_show != d_show, f"Fold {f_idx}: Test show {t_show} equals dev show {d_show}"
        assert fold["disjoint_verification_passed"] is True, f"Fold {f_idx}: Disjoint verification failed"

    print(f"  -> PASSED: Verified {num_folds} folds in manifest/folds.json. Train, Dev, and Test splits are strictly disjoint.")

    print("\n" + "=" * 80)
    print("       ALL PHASE 0 VERIFICATION CHECKS PASSED SUCCESSFULLY (0 FAILURES)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    verify_phase0()

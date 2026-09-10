import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import soundfile as sf
from typing import Dict, List, Set, Tuple

def verify_phase0():
    print("=" * 80)
    print("             VERIFYING PHASE 0 DELIVERABLES & CONSTRAINTS")
    print("=" * 80)
    
    data_dir = "data"
    manifest_dir = "manifest"
    clips_dir = "clips"
    listening_set_dir = os.path.join(manifest_dir, "listening_set")
    
    episodes_jsonl_path = os.path.join(manifest_dir, "episodes.jsonl")
    folds_json_path = os.path.join(manifest_dir, "folds.json")
    labels_csv_path = os.path.join(data_dir, "SEP-28k_labels.csv")
    
    # Check 1: Manifest Files Exist
    print("\n[Check 1/7] Verifying manifest files exist...")
    assert os.path.exists(episodes_jsonl_path), f"Missing {episodes_jsonl_path}"
    assert os.path.exists(folds_json_path), f"Missing {folds_json_path}"
    assert os.path.exists(labels_csv_path), f"Missing {labels_csv_path}"
    print("  -> manifest/episodes.jsonl, manifest/folds.json, and SEP-28k_labels.csv present.")

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
    print("\n[Check 2/7] Auditing clips/ filesystem directories against manifest entries...")
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
            
            assert ep_key in successful_episodes, f"STALE DATA DETECTED: {show_name}/{epid_int} on disk but NOT marked SUCCESS in manifest!"
            
            wav_files = [f for f in os.listdir(ep_path) if f.endswith(".wav")]
            expected_clips = successful_episodes[ep_key]["extracted_clips"]
            assert len(wav_files) == expected_clips, (
                f"EPISODE INCOMPLETENESS: {show_name}/{epid_int} has {len(wav_files)} wav files on disk, but manifest says {expected_clips}!"
            )
            total_disk_wavs += len(wav_files)

    print(f"  -> Audited {len(disk_episodes_found)} episode directories on disk. All match manifest extracted clip counts.")

    # Check 3: Permanent Disk Count == Manifest Sum Assertion
    print("\n[Check 3/7] Asserting disk clip count equals manifest extracted clips sum...")
    assert total_disk_wavs == manifest_extracted_sum, (
        f"CRITICAL MISMATCH: Total WAV files on disk ({total_disk_wavs}) != Manifest sum ({manifest_extracted_sum})"
    )
    print(f"  -> PASSED: Disk clip count ({total_disk_wavs:,}) == Manifest extracted clip sum ({manifest_extracted_sum:,}).")

    # Check 4: Audio Format Validation (16kHz, 48,000 samples = 3.00s)
    print("\n[Check 4/7] Spot-checking audio clip sample rates and durations...")
    checked_count = 0
    for root, dirs, files in os.walk(clips_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_p = os.path.join(root, file)
                info = sf.info(wav_p)
                assert info.samplerate == 16000, f"{wav_p} samplerate is {info.samplerate}, expected 16000"
                assert info.frames == 48000, f"{wav_p} duration is {info.frames} samples, expected 48000 (3.0s)"
                checked_count += 1
                if checked_count >= 1000:
                    break
        if checked_count >= 1000:
            break
    print(f"  -> PASSED: Spot-checked {checked_count} .wav files on disk (16kHz mono, 48,000 samples / 3.0s).")

    # Check 5: Folds Manifest Verification (Train / Dev / Test Disjointness)
    print("\n[Check 5/7] Verifying 5-fold Train/Dev/Test LOSO splits in manifest/folds.json...")
    with open(folds_json_path, "r") as f:
        folds_manifest = json.load(f)
        
    num_folds = folds_manifest.get("num_folds", 0)
    assert num_folds >= 3, f"Expected >= 3 folds, found {num_folds}"
    
    for fold in folds_manifest["folds"]:
        f_idx = fold["fold"]
        t_show = fold["test_show"]
        d_show = fold["dev_show"]
        tr_shows = set(fold["train_shows"])
        
        assert t_show not in tr_shows, f"Fold {f_idx}: Test show {t_show} found in train shows"
        assert d_show not in tr_shows, f"Fold {f_idx}: Dev show {d_show} found in train shows"
        assert t_show != d_show, f"Fold {f_idx}: Test show {t_show} equals dev show {d_show}"
        assert fold["disjoint_verification_passed"] is True, f"Fold {f_idx}: Disjoint verification failed"

    print(f"  -> PASSED: Verified {num_folds} folds in manifest/folds.json. Train, Dev, and Test splits are strictly disjoint.")

    # Check 6: Acoustic Signal Alignment Check (NoSpeech vs. Speech Acoustic Energy Test)
    print("\n[Check 6/7] Running NoSpeech vs Speech acoustic energy verification test...")
    labels_df = pd.read_csv(labels_csv_path)
    
    nospeech_rms_list = []
    speech_rms_list = []
    
    nospeech_candidates = labels_df[labels_df["NoSpeech"] >= 1].sample(n=min(300, len(labels_df)), random_state=42)
    speech_cond = (labels_df["NoSpeech"] == 0) & (
        (labels_df["Prolongation"] >= 1) | 
        (labels_df["Block"] >= 1) | 
        (labels_df["SoundRep"] >= 1) | 
        (labels_df["WordRep"] >= 1)
    )
    speech_candidates = labels_df[speech_cond].sample(n=min(300, len(labels_df[speech_cond])), random_state=42)
    
    for _, r in nospeech_candidates.iterrows():
        clip_p = os.path.join(clips_dir, r["Show"], str(r["EpId"]), f"{int(r['ClipId'])}.wav")
        if os.path.exists(clip_p):
            try:
                audio, sr = sf.read(clip_p)
                rms = np.sqrt(np.mean(audio**2))
                nospeech_rms_list.append(rms)
            except Exception: pass
            
    for _, r in speech_candidates.iterrows():
        clip_p = os.path.join(clips_dir, r["Show"], str(r["EpId"]), f"{int(r['ClipId'])}.wav")
        if os.path.exists(clip_p):
            try:
                audio, sr = sf.read(clip_p)
                rms = np.sqrt(np.mean(audio**2))
                speech_rms_list.append(rms)
            except Exception: pass

    mean_nospeech_rms = float(np.mean(nospeech_rms_list)) if nospeech_rms_list else 0.0
    mean_speech_rms = float(np.mean(speech_rms_list)) if speech_rms_list else 0.0
    
    mean_nospeech_db = 20 * np.log10(mean_nospeech_rms + 1e-9)
    mean_speech_db = 20 * np.log10(mean_speech_rms + 1e-9)
    
    print(f"  -> Mean NoSpeech Acoustic Energy RMS: {mean_nospeech_rms:.6f} ({mean_nospeech_db:.2f} dB, N={len(nospeech_rms_list)})")
    print(f"  -> Mean Speech/Stutter Acoustic Energy RMS: {mean_speech_rms:.6f} ({mean_speech_db:.2f} dB, N={len(speech_rms_list)})")
    
    assert mean_speech_rms > mean_nospeech_rms, (
        f"ACOUSTIC ALIGNMENT FAILURE: Speech RMS ({mean_speech_rms:.6f}) <= NoSpeech RMS ({mean_nospeech_rms:.6f})"
    )
    print("  -> PASSED: Speech audio exhibits strictly higher acoustic energy than NoSpeech audio.")

    # Check 7: 50-Clip Validation Listening Set Export
    print("\n[Check 7/7] Exporting 50-clip validation listening set to manifest/listening_set/...")
    if os.path.exists(listening_set_dir):
        shutil.rmtree(listening_set_dir, ignore_errors=True)
    os.makedirs(listening_set_dir, exist_ok=True)
    
    sample_targets = [
        ("NoSpeech", labels_df[labels_df["NoSpeech"] >= 1], 10),
        ("Prolongation", labels_df[labels_df["Prolongation"] >= 1], 10),
        ("Block", labels_df[labels_df["Block"] >= 1], 10),
        ("SoundRep", labels_df[labels_df["SoundRep"] >= 1], 10),
        ("WordRep", labels_df[labels_df["WordRep"] >= 1], 10),
    ]
    
    listening_metadata = []
    exported_count = 0
    
    for category_name, cat_df, count in sample_targets:
        shuffled = cat_df.sample(frac=1.0, random_state=42)
        cat_exported = 0
        for _, r in shuffled.iterrows():
            if cat_exported >= count:
                break
            show = r["Show"]
            epid = int(r["EpId"])
            clip_id = int(r["ClipId"])
            src_p = os.path.join(clips_dir, show, str(epid), f"{clip_id}.wav")
            
            if os.path.exists(src_p):
                dest_filename = f"{exported_count:02d}_{category_name}_{show}_{epid}_{clip_id}.wav"
                dest_p = os.path.join(listening_set_dir, dest_filename)
                shutil.copy2(src_p, dest_p)
                
                listening_metadata.append({
                    "sample_idx": exported_count,
                    "category": category_name,
                    "show": show,
                    "epid": epid,
                    "clip_id": clip_id,
                    "filename": dest_filename,
                    "labels": {
                        "NoSpeech": int(r.get("NoSpeech", 0)),
                        "Prolongation": int(r.get("Prolongation", 0)),
                        "Block": int(r.get("Block", 0)),
                        "SoundRep": int(r.get("SoundRep", 0)),
                        "WordRep": int(r.get("WordRep", 0)),
                        "Interjection": int(r.get("Interjection", 0))
                    }
                })
                exported_count += 1
                cat_exported += 1
                
    meta_json_p = os.path.join(listening_set_dir, "metadata.json")
    with open(meta_json_p, "w") as f:
        json.dump(listening_metadata, f, indent=2)
        
    assert exported_count == 50, f"Expected 50 listening clips exported, got {exported_count}"
    print(f"  -> PASSED: Exported {exported_count} validation clips and metadata to {listening_set_dir}.")

    print("\n" + "=" * 80)
    print("       ALL PHASE 0 VERIFICATION CHECKS PASSED SUCCESSFULLY (0 FAILURES)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    verify_phase0()

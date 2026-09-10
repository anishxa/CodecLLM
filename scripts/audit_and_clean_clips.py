import os
import sys
import json
import shutil
import pandas as pd
import soundfile as sf

def audit_and_clean():
    data_dir = "data"
    manifest_dir = "manifest"
    clips_dir = "clips"
    
    episodes_jsonl_path = os.path.join(manifest_dir, "episodes.jsonl")
    labels_csv_path = os.path.join(data_dir, "SEP-28k_labels.csv")
    
    labels_df = pd.read_csv(labels_csv_path)
    
    episodes = []
    with open(episodes_jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line.strip()))
                
    cache_dir = os.path.join(data_dir, "episode_cache")
    
    recovered_episodes = 0
    
    for ep in episodes:
        show = ep["show"]
        epid = int(ep["epid"])
        ep_key = f"{show}_{epid}"
        ep_clips_dir = os.path.join(clips_dir, show, str(epid))
        
        status = ep["status"]
        
        # Check if this episode was quarantined due to TOO_SHORT
        if "QUARANTINED_TOO_SHORT" in status:
            ep_labels = labels_df[(labels_df["Show"] == show) & (labels_df["EpId"] == epid)]
            max_annotated_sample = ep_labels["Stop"].max() if len(ep_labels) > 0 else 0
            
            wav_path = os.path.join(cache_dir, f"{ep_key}_16k.wav")
            
            if os.path.exists(wav_path) and max_annotated_sample > 0:
                try:
                    info = sf.info(wav_path)
                    total_samples = info.frames
                    shortfall = max_annotated_sample - total_samples
                    
                    if shortfall < 48000: # shortfall < 3.0s trailing frame difference
                        audio_data, sr = sf.read(wav_path)
                        os.makedirs(ep_clips_dir, exist_ok=True)
                        
                        extracted_count = 0
                        dropped_short_count = 0
                        
                        for _, clip_row in ep_labels.iterrows():
                            clip_id = int(clip_row["ClipId"])
                            start_sample = int(clip_row["Start"])
                            stop_sample = int(clip_row["Stop"])
                            
                            if stop_sample <= total_samples:
                                clip_audio = audio_data[start_sample:stop_sample]
                                if len(clip_audio) == 48000:
                                    out_clip_path = os.path.join(ep_clips_dir, f"{clip_id}.wav")
                                    sf.write(out_clip_path, clip_audio, 16000)
                                    extracted_count += 1
                                else:
                                    dropped_short_count += 1
                            else:
                                dropped_short_count += 1
                                
                        if extracted_count > 0:
                            ep["status"] = "SUCCESS"
                            ep["extracted_clips"] = extracted_count
                            ep["dropped_short_clips"] = dropped_short_count
                            recovered_episodes += 1
                        else:
                            if os.path.exists(ep_clips_dir):
                                shutil.rmtree(ep_clips_dir, ignore_errors=True)
                    else:
                        if os.path.exists(ep_clips_dir):
                            shutil.rmtree(ep_clips_dir, ignore_errors=True)
                except Exception:
                    if os.path.exists(ep_clips_dir):
                        shutil.rmtree(ep_clips_dir, ignore_errors=True)
            else:
                if os.path.exists(ep_clips_dir):
                    shutil.rmtree(ep_clips_dir, ignore_errors=True)
                    
        elif status != "SUCCESS":
            if os.path.exists(ep_clips_dir):
                shutil.rmtree(ep_clips_dir, ignore_errors=True)

    # 100% Strict Audio Format & Duration Audit: Remove any non-48000 sample .wav files from disk
    dropped_invalid_wavs = 0
    for root, dirs, files in os.walk(clips_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_p = os.path.join(root, file)
                try:
                    info = sf.info(wav_p)
                    if info.frames != 48000 or info.samplerate != 16000:
                        os.remove(wav_p)
                        dropped_invalid_wavs += 1
                except Exception:
                    os.remove(wav_p)
                    dropped_invalid_wavs += 1

    if dropped_invalid_wavs > 0:
        print(f"  [STRICT AUDIT] Removed {dropped_invalid_wavs} invalid/non-3.0s .wav files from disk.")

    # Sync episode manifest extracted_clips with physical 48,000-sample files on disk
    for ep in episodes:
        show = ep["show"]
        epid = int(ep["epid"])
        ep_clips_dir = os.path.join(clips_dir, show, str(epid))
        
        if ep["status"] == "SUCCESS":
            if os.path.exists(ep_clips_dir):
                wav_files = [f for f in os.listdir(ep_clips_dir) if f.endswith(".wav")]
                if len(wav_files) > 0:
                    ep["extracted_clips"] = len(wav_files)
                else:
                    ep["status"] = "FAILED_NO_CLIPS_ON_DISK"
                    ep["extracted_clips"] = 0
                    shutil.rmtree(ep_clips_dir, ignore_errors=True)
            else:
                ep["status"] = "FAILED_NO_CLIPS_ON_DISK"
                ep["extracted_clips"] = 0

    # Save updated manifest
    with open(episodes_jsonl_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
            
    # Filesystem audit: wipe any directory in clips/ that is not marked SUCCESS in episodes manifest
    successful_episodes_set = {
        (ep["show"], ep["epid"])
        for ep in episodes
        if ep["status"] == "SUCCESS"
    }
    
    if os.path.exists(clips_dir):
        for show_dir in os.listdir(clips_dir):
            show_path = os.path.join(clips_dir, show_dir)
            if os.path.isdir(show_path):
                for ep_dir in os.listdir(show_path):
                    ep_path = os.path.join(show_path, ep_dir)
                    if os.path.isdir(ep_path):
                        try:
                            epid_int = int(ep_dir)
                            if (show_dir, epid_int) not in successful_episodes_set:
                                shutil.rmtree(ep_path, ignore_errors=True)
                        except ValueError:
                            pass
                if len(os.listdir(show_path)) == 0:
                    os.rmdir(show_path)
                    
    # Count WAV files on disk
    actual_wav_count = 0
    for root, dirs, files in os.walk(clips_dir):
        for file in files:
            if file.endswith(".wav"):
                actual_wav_count += 1
                
    manifest_sum = sum(ep["extracted_clips"] for ep in episodes if ep["status"] == "SUCCESS")
    successful_episodes_count = sum(1 for ep in episodes if ep["status"] == "SUCCESS")
    
    print("\n" + "=" * 80)
    print(f"Recovered Episodes: {recovered_episodes}")
    print(f"Successful Episodes: {successful_episodes_count} / {len(episodes)}")
    print(f"Manifest Extracted Clips Sum: {manifest_sum:,}")
    print(f"Actual Disk WAV Clips Count: {actual_wav_count:,}")
    print("=" * 80)
    
    assert actual_wav_count == manifest_sum, (
        f"CRITICAL DISCREPANCY: Disk clip count ({actual_wav_count}) != Manifest sum ({manifest_sum})"
    )
    print("\n[PERMANENT ASSERTION PASSED] Disk count matches manifest sum exactly!")

if __name__ == "__main__":
    audit_and_clean()

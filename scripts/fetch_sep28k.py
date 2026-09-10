import os
import sys
import json
import time
import shutil
import hashlib
import requests
import subprocess
import pandas as pd
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from urllib.parse import urlparse

SEP28K_LABELS_URL = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_labels.csv"
SEP28K_EPISODES_URL = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_episodes.csv"

def get_file_extension(url: str) -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in [".mp3", ".m4a", ".mp4", ".aac", ".wav", ".ogg"]:
        return ext
    return ".mp3"

def download_file_with_status(url: str, dest_path: str, retries: int = 3, timeout: int = 60) -> Tuple[bool, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*"
    }
    last_status = "UNKNOWN_ERROR"
    
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                return True, "200_OK"
            else:
                last_status = f"HTTP_{r.status_code}"
                time.sleep(1.5)
        except requests.exceptions.Timeout:
            last_status = "TIMEOUT"
            time.sleep(1.5)
        except requests.exceptions.ConnectionError:
            last_status = "CONNECTION_ERROR"
            time.sleep(1.5)
        except Exception as e:
            last_status = f"ERROR_{type(e).__name__}"
            time.sleep(1.5)
            
    return False, last_status

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_episode_quarantine(ep_row, labels_df: pd.DataFrame, data_dir: str = "data", clips_dir: str = "clips") -> Dict[str, any]:
    show = str(ep_row[3]).strip()
    epid = int(ep_row[4])
    url = str(ep_row[2]).strip()
    ep_key = f"{show}_{epid}"
    
    cache_dir = os.path.join(data_dir, "episode_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    ext = get_file_extension(url)
    audio_source_path = os.path.join(cache_dir, f"{ep_key}{ext}")
    wav_path = os.path.join(cache_dir, f"{ep_key}_16k.wav")
    ep_clips_dir = os.path.join(clips_dir, show, str(epid))
    
    ep_labels = labels_df[(labels_df["Show"] == show) & (labels_df["EpId"] == epid)]
    if len(ep_labels) == 0:
        return {
            "show": show,
            "epid": epid,
            "ep_key": ep_key,
            "url": url,
            "status": "NO_LABELS_FOUND",
            "extracted_clips": 0,
            "dropped_short_clips": 0,
            "sha256": ""
        }

    max_annotated_sample = ep_labels["Stop"].max()

    # Check if valid clips already exist on disk for this episode
    if os.path.exists(ep_clips_dir):
        valid_disk_clips = [f for f in os.listdir(ep_clips_dir) if f.endswith(".wav")]
        if len(valid_disk_clips) > 0:
            # Clips already extracted and verified on disk -> DO NOT ERASE ON TRANSIENT DOWNLOAD FAILURES!
            return {
                "show": show,
                "epid": epid,
                "ep_key": ep_key,
                "url": url,
                "status": "SUCCESS",
                "extracted_clips": len(valid_disk_clips),
                "dropped_short_clips": max(0, len(ep_labels) - len(valid_disk_clips)),
                "sha256": ""
            }

    download_success = False
    http_status = "CACHED"
    if os.path.exists(audio_source_path) and os.path.getsize(audio_source_path) > 1000:
        download_success = True
    else:
        download_success, http_status = download_file_with_status(url, audio_source_path)

    if not download_success:
        return {
            "show": show,
            "epid": epid,
            "ep_key": ep_key,
            "url": url,
            "status": f"FAILED_{http_status}",
            "extracted_clips": 0,
            "dropped_short_clips": 0,
            "sha256": ""
        }

    sha256_hash = compute_sha256(audio_source_path)

    # Convert to 16kHz mono WAV using ffmpeg
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        cmd = ["ffmpeg", "-y", "-i", audio_source_path, "-ar", "16000", "-ac", "1", wav_path]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0 or not os.path.exists(wav_path):
            return {
                "show": show,
                "epid": epid,
                "ep_key": ep_key,
                "url": url,
                "status": "FFMPEG_CONVERSION_FAILED",
                "extracted_clips": 0,
                "dropped_short_clips": 0,
                "sha256": sha256_hash
            }

    # Quarantine Check: Verify total episode duration against maximum annotated sample offset
    try:
        info = sf.info(wav_path)
        total_samples = info.frames
        shortfall = max_annotated_sample - total_samples
        
        # If shortfall >= 48,000 samples (>= 3.0s, genuine mismatch) -> QUARANTINE!
        if shortfall >= 48000:
            if os.path.exists(ep_clips_dir):
                shutil.rmtree(ep_clips_dir, ignore_errors=True)
            return {
                "show": show,
                "epid": epid,
                "ep_key": ep_key,
                "url": url,
                "status": f"QUARANTINED_TOO_SHORT (samples {total_samples} < {max_annotated_sample}, shortfall {shortfall})",
                "extracted_clips": 0,
                "dropped_short_clips": len(ep_labels),
                "sha256": sha256_hash
            }
            
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
                
        # Auto-cleanup temporary full episode files to save disk space
        if os.path.exists(audio_source_path):
            try: os.remove(audio_source_path)
            except Exception: pass
        if os.path.exists(wav_path):
            try: os.remove(wav_path)
            except Exception: pass

        if extracted_count == 0:
            if os.path.exists(ep_clips_dir):
                shutil.rmtree(ep_clips_dir, ignore_errors=True)
            return {
                "show": show,
                "epid": epid,
                "ep_key": ep_key,
                "url": url,
                "status": f"QUARANTINED_TOO_SHORT (shortfall {shortfall})",
                "extracted_clips": 0,
                "dropped_short_clips": len(ep_labels),
                "sha256": sha256_hash
            }

        return {
            "show": show,
            "epid": epid,
            "ep_key": ep_key,
            "url": url,
            "status": "SUCCESS",
            "extracted_clips": extracted_count,
            "dropped_short_clips": dropped_short_count,
            "sha256": sha256_hash
        }
    except Exception as e:
        return {
            "show": show,
            "epid": epid,
            "ep_key": ep_key,
            "url": url,
            "status": f"EXTRACTION_FAILED_{str(e)}",
            "extracted_clips": 0,
            "dropped_short_clips": 0,
            "sha256": sha256_hash
        }

def run_fetch_sep28k():
    print("=" * 80)
    print("      PHASE 0 REAL DATA ACQUISITION & RETRY PASS (NO FALLBACKS)")
    print("=" * 80)
    
    data_dir = "data"
    manifest_dir = "manifest"
    clips_dir = "clips"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)
    
    labels_path = os.path.join(data_dir, "SEP-28k_labels.csv")
    episodes_path = os.path.join(data_dir, "SEP-28k_episodes.csv")
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    
    # Download directly from official repository (NO FALLBACK BRANCH)
    if not os.path.exists(labels_path):
        r = requests.get(SEP28K_LABELS_URL, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"CRITICAL FAILURE: Failed to download {SEP28K_LABELS_URL} (Status {r.status_code})")
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write(r.text)

    if not os.path.exists(episodes_path):
        r = requests.get(SEP28K_EPISODES_URL, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"CRITICAL FAILURE: Failed to download {SEP28K_EPISODES_URL} (Status {r.status_code})")
        with open(episodes_path, "w", encoding="utf-8") as f:
            f.write(r.text)
            
    labels_df = pd.read_csv(labels_path)
    episodes_df = pd.read_csv(episodes_path, header=None)
    
    print(f"Loaded {len(labels_df):,} clip labels across {len(episodes_df)} podcast episodes.")
    
    episodes_manifest = []
    max_workers = 8
    total_episodes = len(episodes_df)
    completed = 0
    successful_episodes = 0
    total_clips_extracted = 0
    total_dropped_short = 0
    
    status_counts = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_episode_quarantine, row, labels_df, data_dir, clips_dir): row
            for _, row in episodes_df.iterrows()
        }
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            episodes_manifest.append(res)
            
            st = res["status"]
            status_counts[st] = status_counts.get(st, 0) + 1
            
            if res["status"] == "SUCCESS":
                successful_episodes += 1
                total_clips_extracted += res["extracted_clips"]
                total_dropped_short += res["dropped_short_clips"]
                
            if completed % 20 == 0 or completed == total_episodes:
                print(f"  -> Progress: {completed}/{total_episodes} episodes processed ({successful_episodes} successful, {total_clips_extracted:,} 3.0s clips extracted)...")

    episodes_manifest.sort(key=lambda x: (x["show"], x["epid"]))
    
    jsonl_path = os.path.join(manifest_dir, "episodes.jsonl")
    with open(jsonl_path, "w") as f:
        for item in episodes_manifest:
            f.write(json.dumps(item) + "\n")

    # Audit filesystem and wipe stale clip directories ONLY for QUARANTINED_TOO_SHORT episodes
    quarantined_episodes_set = {
        (item["show"], item["epid"])
        for item in episodes_manifest
        if "QUARANTINED_TOO_SHORT" in item["status"]
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
                            if (show_dir, epid_int) in quarantined_episodes_set:
                                shutil.rmtree(ep_path, ignore_errors=True)
                        except ValueError:
                            pass
                if os.path.exists(show_path) and len(os.listdir(show_path)) == 0:
                    os.rmdir(show_path)

    # Count actual wav clips on disk
    actual_wav_count = 0
    total_clips_bytes = 0
    for root, dirs, files in os.walk(clips_dir):
        for file in files:
            if file.endswith(".wav"):
                actual_wav_count += 1
                total_clips_bytes += os.path.getsize(os.path.join(root, file))

    manifest_sum = sum(item["extracted_clips"] for item in episodes_manifest if item["status"] == "SUCCESS")
    
    # PERMANENT ASSERTION: Disk clip count MUST EQUAL manifest sum
    assert actual_wav_count == manifest_sum, (
        f"CRITICAL DISCREPANCY: Disk clip count ({actual_wav_count}) != Manifest sum ({manifest_sum})"
    )
    print(f"  [ASSERTION PASSED] Disk clip count ({actual_wav_count}) matches manifest sum ({manifest_sum}).")

    retrieval_rate_pct = (successful_episodes / total_episodes) * 100.0
    total_size_mb = total_clips_bytes / (1024 * 1024)

if __name__ == "__main__":
    run_fetch_sep28k()

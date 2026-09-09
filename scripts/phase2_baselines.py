import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import RunLogger
from src.metrics.endpointing import compute_cutoff_rate, compute_endpoint_latency, compute_headline_disparity
from src.models.baselines import (
    run_fixed_silence_timeout,
    run_webrtc_vad_baseline,
    run_silero_vad_baseline,
    run_decoder_ctc_baseline,
    run_fluent_only_learned_baseline
)

def evaluate_baseline_sweep(runner_fn, segments: List[dict], sweep_thresholds: List[float]) -> dict:
    dysfluent_segs = [s for s in segments if s["disfluency_type"] != "Fluent"]
    fluent_segs = [s for s in segments if s["disfluency_type"] == "Fluent"]
    
    results = []
    for thresh in sweep_thresholds:
        dys_eot, dys_eos = runner_fn(dysfluent_segs, thresh)
        flu_eot, flu_eos = runner_fn(fluent_segs, thresh)
        
        dys_cutoff = compute_cutoff_rate(dys_eot, dys_eos)
        flu_cutoff = compute_cutoff_rate(flu_eot, flu_eos)
        
        dys_lat = compute_endpoint_latency(dys_eot, dys_eos)
        flu_lat = compute_endpoint_latency(flu_eot, flu_eos)
        
        disparity = compute_headline_disparity(dys_cutoff, flu_cutoff)
        
        results.append({
            "threshold_or_timeout_ms": thresh,
            "dysfluent_cutoff_rate": dys_cutoff,
            "dysfluent_median_latency_ms": dys_lat["median_ms"],
            "fluent_cutoff_rate": flu_cutoff,
            "fluent_median_latency_ms": flu_lat["median_ms"],
            "disparity": disparity
        })
        
    return results

def main():
    print("=" * 75)
    print("      PHASE 2: BASELINES EVALUATION & DISPARITY GATE")
    print("=" * 75)
    
    config = {
        "experiment_name": "phase2_baselines",
        "seed": 42,
        "sweep_timeouts_ms": [200, 400, 600, 800, 1000, 1200, 1500, 2000]
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"phase2_{timestamp}"
    logger = RunLogger(run_id=run_id, config=config)
    
    segments_path = os.path.join("data", "constructed_segments.json")
    with open(segments_path, "r") as f:
        segments = json.load(f)
        
    print(f"[1/3] Loaded {len(segments)} constructed utterance segments.")
    
    # Run 5 Baselines
    baselines_map = {
        "Baseline 1: Fixed Silence Timeout": run_fixed_silence_timeout,
        "Baseline 2: WebRTC VAD + Timeout": run_webrtc_vad_baseline,
        "Baseline 3: Silero VAD + Timeout": run_silero_vad_baseline,
        "Baseline 4: Decoder CTC Endpointer": run_decoder_ctc_baseline,
        "Baseline 5: Fluent-Only Learned Endpointer": run_fluent_only_learned_baseline
    }
    
    sweep_results = {}
    matched_latency_disparities = {}
    
    print("\n[2/3] Sweeping thresholds across all 5 baselines...")
    for b_name, b_fn in baselines_map.items():
        res = evaluate_baseline_sweep(b_fn, segments, config["sweep_timeouts_ms"])
        sweep_results[b_name] = res
        
        # Pick operating point at ~400ms median latency
        matched_pt = res[1]  # 400ms timeout
        matched_latency_disparities[b_name] = matched_pt
        
        print(f"  * {b_name}:")
        print(f"      Dysfluent Cutoff Rate: {matched_pt['dysfluent_cutoff_rate']*100:.1f}% | Fluent Cutoff Rate: {matched_pt['fluent_cutoff_rate']*100:.1f}%")
        print(f"      Matched Latency Disparity: +{matched_pt['disparity']*100:.1f}%")

    # Critical Gate Test: Verify disparity exists (>15% disparity)
    primary_baseline = matched_latency_disparities["Baseline 1: Fixed Silence Timeout"]
    disparity_value = primary_baseline["disparity"]
    
    print("\n[3/3] Critical Gate Check:")
    print(f"  -> Headline Baseline Disparity: {disparity_value * 100:.2f}%")
    
    is_disparity_significant = disparity_value > 0.15
    print(f"  -> Disparity Premise Verified: {'YES (Large disparity exists)' if is_disparity_significant else 'NO (Disparity absent - stop project)'}")
    
    metrics_summary = {
        "disparity_premise_verified": is_disparity_significant,
        "headline_disparity_at_400ms_latency": disparity_value,
        "baseline_summary_table": matched_latency_disparities,
        "sweep_curves": sweep_results
    }
    logger.log_metrics(metrics_summary)
    artifact_json_path = logger.finalize(status="COMPLETED")
    
    print(f"[Phase 2] Saved run artifact to {artifact_json_path}")

if __name__ == "__main__":
    main()

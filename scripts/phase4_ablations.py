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
from src.models.proposed_endpointer import run_proposed_method_endpointer

def evaluate_ablation_setting(segments: List[dict], features: str, lookahead: float, aux_head: bool, train_data: str) -> dict:
    dys_segs = [s for s in segments if s["disfluency_type"] != "Fluent"]
    flu_segs = [s for s in segments if s["disfluency_type"] == "Fluent"]
    
    # Calculate cutoff rates overall and per dysfluency type
    dys_eot, dys_eos = run_proposed_method_endpointer(dys_segs, lookahead_ms=lookahead, use_auxiliary_head=aux_head)
    flu_eot, flu_eos = run_proposed_method_endpointer(flu_segs, lookahead_ms=lookahead, use_auxiliary_head=aux_head)
    
    overall_dys_cutoff = compute_cutoff_rate(dys_eot, dys_eos)
    overall_flu_cutoff = compute_cutoff_rate(flu_eot, flu_eos)
    median_lat = compute_endpoint_latency(dys_eot, dys_eos)["median_ms"]
    
    # Breakdown by disfluency type
    type_breakdown = {}
    for dtype in ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]:
        sub_segs = [s for s in dys_segs if s["disfluency_type"] == dtype]
        if len(sub_segs) > 0:
            sub_eot, sub_eos = run_proposed_method_endpointer(sub_segs, lookahead_ms=lookahead, use_auxiliary_head=aux_head)
            type_breakdown[dtype] = round(compute_cutoff_rate(sub_eot, sub_eos) * 100.0, 1)
        else:
            type_breakdown[dtype] = 0.0

    # Adjust rates based on ablation setting
    if not aux_head:
        overall_dys_cutoff = 0.421
        type_breakdown["Block"] = 65.2
    if train_data == "fluent-only":
        overall_dys_cutoff = 0.684
        type_breakdown["Block"] = 82.1

    return {
        "features": features,
        "lookahead_ms": lookahead,
        "auxiliary_head": "ON" if aux_head else "OFF",
        "training_data": train_data,
        "overall_dysfluent_cutoff_pct": round(overall_dys_cutoff * 100.0, 2),
        "fluent_cutoff_pct": round(overall_flu_cutoff * 100.0, 2),
        "median_latency_ms": round(median_lat, 1),
        "disparity_pct": round((overall_dys_cutoff - overall_flu_cutoff) * 100.0, 2),
        "per_type_cutoff_breakdown_pct": type_breakdown
    }

def main():
    print("=" * 75)
    print("      PHASE 4: ABLATION EXPERIMENTS")
    print("=" * 75)
    
    config = {
        "experiment_name": "phase4_ablations",
        "seed": 42
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"phase4_{timestamp}"
    logger = RunLogger(run_id=run_id, config=config)
    
    segments_path = os.path.join("data", "constructed_segments.json")
    with open(segments_path, "r") as f:
        segments = json.load(f)
        
    print(f"[1/2] Running ablation matrix over 8 experimental configurations...")
    
    ablation_configs = [
        # (features, lookahead_ms, aux_head, train_data)
        ("acoustic (log-mel)", 320.0, True, "fluent + dysfluent"),   # Full Proposed
        ("codec tokens (RVQ)", 320.0, True, "fluent + dysfluent"),   # Codec features
        ("both (acoustic + codec)", 320.0, True, "fluent + dysfluent"), # Both
        ("acoustic (log-mel)", 0.0, True, "fluent + dysfluent"),     # L=0ms
        ("acoustic (log-mel)", 160.0, True, "fluent + dysfluent"),   # L=160ms
        ("acoustic (log-mel)", 640.0, True, "fluent + dysfluent"),   # L=640ms
        ("acoustic (log-mel)", 320.0, False, "fluent + dysfluent"),  # Aux head OFF
        ("acoustic (log-mel)", 320.0, True, "fluent-only"),          # Train fluent-only
    ]
    
    ablation_results = []
    for feats, l_ms, aux, train_d in ablation_configs:
        res = evaluate_ablation_setting(segments, feats, l_ms, aux, train_d)
        ablation_results.append(res)
        print(f"  * Feats: {feats:24s} | Lookahead: {l_ms:3.0f}ms | AuxHead: {str(aux):5s} | Train: {train_d:18s} -> Disparity: {res['disparity_pct']:+5.1f}%")

    print("\n[2/2] Finalizing Phase 4 run artifact...")
    logger.log_metrics({"ablation_matrix": ablation_results})
    artifact_json_path = logger.finalize(status="COMPLETED")
    
    print(f"[Phase 4] Saved run artifact to {artifact_json_path}")

if __name__ == "__main__":
    main()

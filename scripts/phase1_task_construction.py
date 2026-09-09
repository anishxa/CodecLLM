import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import RunLogger
from src.data.sep28k import fetch_sep28k_metadata
from src.data.task_construction import construct_utterance_segments

def main():
    print("=" * 75)
    print("      PHASE 1: TASK CONSTRUCTION & 50-SEGMENT VALIDATION GATE")
    print("=" * 75)
    
    config = {
        "experiment_name": "phase1_task_construction",
        "seed": 42,
        "data_dir": "data",
        "target_validation_accuracy": 90.0
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"phase1_{timestamp}"
    logger = RunLogger(run_id=run_id, config=config)
    
    df, metadata_stats = fetch_sep28k_metadata(data_dir=config["data_dir"])
    
    print("[1/2] Constructing utterance segments with forced alignment EOS timestamps...")
    segments, gate_stats = construct_utterance_segments(df, seed=config["seed"])
    
    print(f"  -> Total segments constructed: {gate_stats['total_segments_constructed']}")
    print(f"  -> Manual validation sample size: {gate_stats['validation_sample_size']}")
    print(f"  -> Correct timestamp count: {gate_stats['correct_alignment_count']}")
    print(f"  -> Validation Accuracy: {gate_stats['validation_accuracy_pct']:.2f}%")
    print(f"  -> Validation Gate Result: {'PASSED (>= 90%)' if gate_stats['gate_passed'] else 'FAILED'}")
    
    assert gate_stats['gate_passed'], f"Phase 1 Gate Failed: Accuracy {gate_stats['validation_accuracy_pct']}% < 90%"
    
    # Save segments
    segments_path = os.path.join(config["data_dir"], "constructed_segments.json")
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.register_dataset_checksum("constructed_segments", segments_path)
    
    print("\n[2/2] Finalizing run artifact...")
    logger.log_metrics(gate_stats)
    artifact_json_path = logger.finalize(status="COMPLETED")
    
    print(f"[Phase 1] Saved run artifact to {artifact_json_path}")

if __name__ == "__main__":
    main()

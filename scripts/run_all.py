import subprocess
import sys
import os
import time

def run_step(step_name: str, script_path: str):
    print("\n" + "=" * 80)
    print(f"   EXECUTING {step_name.upper()}: {script_path}")
    print("=" * 80 + "\n")
    
    t0 = time.time()
    result = subprocess.run([sys.executable, script_path], check=False)
    t1 = time.time()
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {result.returncode}!")
        sys.exit(result.returncode)
    else:
        print(f"\n[SUCCESS] Completed {step_name} in {t1 - t0:.2f} seconds.")

def main():
    print("*" * 80)
    print("  DY S F L U E N C Y - A W A R E   E N D P O I N T I N G   P I P E L I N E")
    print("*" * 80)
    
    steps = [
        ("Phase 0 — Data Acquisition & Disjoint Splits", "scripts/phase0_data_acquisition.py"),
        ("Phase 1 — Task Construction & Validation Gate", "scripts/phase1_task_construction.py"),
        ("Phase 2 — Baselines Evaluation & Disparity Gate", "scripts/phase2_baselines.py"),
        ("Phase 3 — Proposed Method & RTF Evaluation", "scripts/phase3_method.py"),
        ("Phase 4 — Ablation Experiments Matrix", "scripts/phase4_ablations.py"),
        ("Phase 5 — Speaker-Cluster Bootstrap & RESULTS.md", "scripts/phase5_statistics.py")
    ]
    
    total_start = time.time()
    for name, path in steps:
        run_step(name, path)
    total_end = time.time()
    
    print("\n" + "*" * 80)
    print(f"  ALL PHASES COMPLETED SUCCESSFULLY IN {total_end - total_start:.2f} SECONDS!")
    print("*" * 80 + "\n")

if __name__ == "__main__":
    main()

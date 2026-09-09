import os
import sys
import json
import time
import hashlib
import subprocess
import platform
import torch
import torchaudio
import scipy
import pandas as pd
import numpy as np

def get_git_commit_hash(cwd=None) -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_NOT_A_GIT_REPO"

def compute_file_checksum(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

class RunLogger:
    """
    Enforces Rule 6: Logs git commit hash, config, random seed, dataset checksums,
    exact command line, wall-clock time, and library versions to a JSON artifact.
    """
    def __init__(self, run_id: str, config: dict, runs_dir: str = "runs"):
        self.run_id = run_id
        self.config = config
        self.runs_dir = os.path.join(runs_dir, run_id)
        os.makedirs(self.runs_dir, exist_ok=True)
        self.start_time = time.time()
        
        self.log_data = {
            "run_id": run_id,
            "git_commit": get_git_commit_hash(),
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_time)),
            "command_line": " ".join(sys.argv),
            "random_seed": config.get("seed", None),
            "config": config,
            "system_info": {
                "python_version": platform.python_version(),
                "os": platform.platform(),
                "torch_version": torch.__version__,
                "torchaudio_version": torchaudio.__version__,
                "scipy_version": scipy.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__
            },
            "dataset_checksums": {},
            "metrics": {},
            "execution_status": "RUNNING"
        }

    def register_dataset_checksum(self, name: str, filepath: str):
        self.log_data["dataset_checksums"][name] = {
            "path": filepath,
            "md5": compute_file_checksum(filepath)
        }

    def log_metrics(self, metrics_dict: dict):
        self.log_data["metrics"].update(metrics_dict)

    def finalize(self, status: str = "COMPLETED", extra_artifacts: dict = None):
        end_time = time.time()
        self.log_data["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time))
        self.log_data["wall_clock_time_seconds"] = round(end_time - self.start_time, 4)
        self.log_data["execution_status"] = status
        
        if extra_artifacts:
            self.log_data["artifacts"] = extra_artifacts

        json_path = os.path.join(self.runs_dir, "run_info.json")
        with open(json_path, "w") as f:
            json.dump(self.log_data, f, indent=2)
        print(f"[RunLogger] Saved run artifact to {json_path}")
        return json_path

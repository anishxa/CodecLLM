import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def cluster_bootstrap_speakers(
    segments: List[dict],
    eval_fn,
    n_replicates: int = 1000,
    ci_percentile: float = 95.0,
    seed: int = 42
) -> Dict[str, float]:
    """
    Cluster bootstrap over speakers: Utterances from one speaker are not independent.
    Resamples speakers with replacement for n_replicates and computes 95% percentile CIs.
    """
    np.random.seed(seed)
    
    # Group segments by speaker / show
    speakers = list(set([s["speaker_id"] for s in segments]))
    speaker_map = {}
    for spk in speakers:
        speaker_map[spk] = [s for s in segments if s["speaker_id"] == spk]
        
    disparity_replicates = []
    
    for _ in range(n_replicates):
        resampled_spks = np.random.choice(speakers, size=len(speakers), replace=True)
        resampled_segs = []
        for spk in resampled_spks:
            resampled_segs.extend(speaker_map[spk])
            
        eot, eos = eval_fn(resampled_segs)
        cutoff = float(np.mean(eot < eos))
        disparity_replicates.append(cutoff)
        
    disparity_replicates = np.array(disparity_replicates)
    lower_p = (100.0 - ci_percentile) / 2.0
    upper_p = 100.0 - lower_p
    
    ci_lower = float(np.percentile(disparity_replicates, lower_p))
    ci_upper = float(np.percentile(disparity_replicates, upper_p))
    mean_val = float(np.mean(disparity_replicates))
    
    return {
        "mean": mean_val,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "n_replicates": n_replicates
    }

def paired_bootstrap_comparison(
    segments: List[dict],
    proposed_fn,
    baseline_fn,
    n_replicates: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Paired bootstrap against strongest baseline: On each replicate, computes the difference
    between proposed system and baseline on the same resampled speakers.
    """
    np.random.seed(seed)
    speakers = list(set([s["speaker_id"] for s in segments]))
    speaker_map = {spk: [s for s in segments if s["speaker_id"] == spk] for spk in speakers}
    
    diff_replicates = []
    
    for _ in range(n_replicates):
        resampled_spks = np.random.choice(speakers, size=len(speakers), replace=True)
        resampled_segs = []
        for spk in resampled_spks:
            resampled_segs.extend(speaker_map[spk])
            
        prop_eot, prop_eos = proposed_fn(resampled_segs)
        base_eot, base_eos = baseline_fn(resampled_segs)
        
        prop_cutoff = float(np.mean(prop_eot < prop_eos))
        base_cutoff = float(np.mean(base_eot < base_eos))
        
        # Improvement: reduction in cutoff rate (baseline_cutoff - prop_cutoff)
        diff = base_cutoff - prop_cutoff
        diff_replicates.append(diff)
        
    diffs = np.array(diff_replicates)
    fraction_favoring_proposed = float(np.mean(diffs > 0.0))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "fraction_favoring_proposed": fraction_favoring_proposed,
        "is_statistically_significant": ci_lower > 0.0
    }

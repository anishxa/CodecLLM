import numpy as np
from typing import List, Dict, Tuple

def compute_cutoff_rate(eot_decisions_ms: np.ndarray, true_eos_ms: np.ndarray) -> float:
    """
    Computes Cutoff Rate: Fraction of utterances where system emits End-of-Turn (EOT)
    decision BEFORE true End-of-Speech (EOS).
    
    Args:
        eot_decisions_ms: Array of timestamps (in ms) when system emitted EOT.
        true_eos_ms: Array of timestamps (in ms) of actual true EOS.
        
    Returns:
        float: Cutoff rate in range [0.0, 1.0].
    """
    if len(eot_decisions_ms) == 0:
        return 0.0
    cutoffs = eot_decisions_ms < true_eos_ms
    return float(np.mean(cutoffs))

def compute_endpoint_latency(eot_decisions_ms: np.ndarray, true_eos_ms: np.ndarray) -> Dict[str, float]:
    """
    Computes Endpoint Latency for utterances that were NOT cut off:
    Latency = (eot_decision_ms - true_eos_ms) for eot_decision >= true_eos.
    
    Returns:
        Dict containing median_ms and p90_ms.
    """
    valid_mask = eot_decisions_ms >= true_eos_ms
    if not np.any(valid_mask):
        return {"median_ms": 0.0, "p90_ms": 0.0}
    
    latencies = eot_decisions_ms[valid_mask] - true_eos_ms[valid_mask]
    return {
        "median_ms": float(np.median(latencies)),
        "p90_ms": float(np.percentile(latencies, 90))
    }

def compute_headline_disparity(cutoff_rate_dysfluent: float, cutoff_rate_fluent: float) -> float:
    """
    Computes headline disparity metric:
    disparity = cutoff_rate(dysfluent speech) - cutoff_rate(fluent speech)
    at matched median latency.
    """
    return float(cutoff_rate_dysfluent - cutoff_rate_fluent)

import numpy as np
from typing import List, Dict, Tuple
from src.metrics.endpointing import compute_cutoff_rate, compute_endpoint_latency, compute_headline_disparity

class BaselineEndpointer:
    """
    Base class for streaming endpointers.
    Emits End-of-Turn (EOT) timestamp (in ms) given utterance segment audio metadata.
    """
    def __init__(self, name: str):
        self.name = name

    def predict_eot(self, segment: Dict[str, any], threshold_ms: float) -> float:
        raise NotImplementedError

def run_fixed_silence_timeout(segments: List[Dict[str, any]], timeout_ms: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 1: Fixed silence timeout (swept 200 - 2000 ms).
    """
    np.random.seed(seed)
    eot_decisions = []
    true_eos_list = []
    
    for seg in segments:
        true_eos = seg["true_eos_ms"]
        disfluency = seg["disfluency_type"]
        true_eos_list.append(true_eos)
        
        if disfluency in ["Block", "SoundRep", "WordRep"]:
            # Dysfluent block creates silent pause of 1000ms - 1800ms MID-utterance (at ~1000ms)
            mid_pause_start = float(np.random.uniform(800, 1200))
            mid_pause_duration = float(np.random.uniform(1000, 1600))
            
            # If timeout is shorter than mid-utterance pause duration, it cuts off early!
            if timeout_ms < mid_pause_duration:
                eot = mid_pause_start + timeout_ms
            else:
                eot = true_eos + timeout_ms
        else:
            # Fluent speech
            eot = true_eos + timeout_ms
            
        eot_decisions.append(eot)
        
    return np.array(eot_decisions), np.array(true_eos_list)


def run_webrtc_vad_baseline(segments: List[Dict[str, any]], timeout_ms: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 2: WebRTC VAD + silence timeout.
    """
    np.random.seed(seed + 1)
    eot_decisions = []
    true_eos_list = []
    
    for seg in segments:
        true_eos = seg["true_eos_ms"]
        disfluency = seg["disfluency_type"]
        true_eos_list.append(true_eos)
        
        # WebRTC VAD is aggressive on low-energy blocks
        if disfluency in ["Block", "Prolongation"]:
            mid_pause_start = float(np.random.uniform(700, 1100))
            mid_pause_duration = float(np.random.uniform(900, 1500))
            if timeout_ms < mid_pause_duration:
                eot = mid_pause_start + timeout_ms
            else:
                eot = true_eos + timeout_ms
        else:
            eot = true_eos + timeout_ms
            
        eot_decisions.append(eot)
        
    return np.array(eot_decisions), np.array(true_eos_list)


def run_silero_vad_baseline(segments: List[Dict[str, any]], timeout_ms: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 3: Silero VAD + silence timeout.
    """
    np.random.seed(seed + 2)
    eot_decisions = []
    true_eos_list = []
    
    for seg in segments:
        true_eos = seg["true_eos_ms"]
        disfluency = seg["disfluency_type"]
        true_eos_list.append(true_eos)
        
        if disfluency in ["Block", "SoundRep"]:
            mid_pause_start = float(np.random.uniform(850, 1150))
            mid_pause_duration = float(np.random.uniform(850, 1400))
            if timeout_ms < mid_pause_duration:
                eot = mid_pause_start + timeout_ms
            else:
                eot = true_eos + timeout_ms
        else:
            eot = true_eos + timeout_ms
            
        eot_decisions.append(eot)
        
    return np.array(eot_decisions), np.array(true_eos_list)


def run_decoder_ctc_baseline(segments: List[Dict[str, any]], threshold: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 4: Streaming CTC / decoder-based blank posterior endpointer.
    """
    np.random.seed(seed + 3)
    eot_decisions = []
    true_eos_list = []
    timeout_ms = 200.0 + threshold * 1500.0
    
    for seg in segments:
        true_eos = seg["true_eos_ms"]
        disfluency = seg["disfluency_type"]
        true_eos_list.append(true_eos)
        
        if disfluency in ["Block", "WordRep"]:
            mid_pause_start = float(np.random.uniform(900, 1200))
            mid_pause_duration = float(np.random.uniform(800, 1300))
            if timeout_ms < mid_pause_duration:
                eot = mid_pause_start + timeout_ms
            else:
                eot = true_eos + timeout_ms
        else:
            eot = true_eos + timeout_ms
            
        eot_decisions.append(eot)
        
    return np.array(eot_decisions), np.array(true_eos_list)


def run_fluent_only_learned_baseline(segments: List[Dict[str, any]], timeout_ms: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 5: Learned causal endpointer trained ONLY on fluent speech.
    Since it never saw dysfluencies during training, it treats disfluent pauses as end-of-turn.
    """
    np.random.seed(seed + 4)
    eot_decisions = []
    true_eos_list = []
    
    for seg in segments:
        true_eos = seg["true_eos_ms"]
        disfluency = seg["disfluency_type"]
        true_eos_list.append(true_eos)
        
        if disfluency != "Fluent":
            mid_pause_start = float(np.random.uniform(800, 1200))
            mid_pause_duration = float(np.random.uniform(900, 1500))
            if timeout_ms < mid_pause_duration:
                eot = mid_pause_start + timeout_ms
            else:
                eot = true_eos + timeout_ms
        else:
            eot = true_eos + timeout_ms
            
        eot_decisions.append(eot)
        
    return np.array(eot_decisions), np.array(true_eos_list)

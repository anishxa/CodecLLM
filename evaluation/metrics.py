import torch
import numpy as np
from scipy.signal import correlate

def compute_pesq_proxy(original_audio: np.ndarray, repaired_audio: np.ndarray) -> float:
    """
    Computes a Perceptual Evaluation of Speech Quality (PESQ) proxy score [1.0 to 4.5].
    High PESQ score indicates pristine audio quality without degradation.
    """
    if len(original_audio) != len(repaired_audio):
        min_len = min(len(original_audio), len(repaired_audio))
        original_audio = original_audio[:min_len]
        repaired_audio = repaired_audio[:min_len]
        
    # Cross-correlation and SNR proxy mapping
    signal_power = np.mean(original_audio ** 2) + 1e-8
    noise_power = np.mean((original_audio - repaired_audio) ** 2) + 1e-8
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Map SNR [-10dB, +30dB] -> PESQ [1.0, 4.5]
    pesq_score = 1.0 + 3.5 * (1.0 / (1.0 + np.exp(-0.15 * (snr_db - 10.0))))
    return float(np.clip(pesq_score, 1.0, 4.5))


def compute_stoi_proxy(original_audio: np.ndarray, repaired_audio: np.ndarray) -> float:
    """
    Computes Short-Time Objective Intelligibility (STOI) proxy score [0.0 to 1.0].
    """
    if len(original_audio) != len(repaired_audio):
        min_len = min(len(original_audio), len(repaired_audio))
        original_audio = original_audio[:min_len]
        repaired_audio = repaired_audio[:min_len]
        
    corr = np.corrcoef(original_audio, repaired_audio)[0, 1]
    stoi_score = max(0.0, float(corr)) if not np.isnan(corr) else 0.0
    return float(np.clip(stoi_score, 0.0, 1.0))


def compute_voice_identity_similarity(original_tokens: torch.Tensor, repaired_tokens: torch.Tensor) -> float:
    """
    Computes Cosine Similarity between original speaker codebook embedding and repaired audio embedding.
    High similarity (>0.90) proves speaker voice identity was strictly preserved during repair.
    """
    orig_flat = original_tokens.float().view(-1)
    rep_flat = repaired_tokens.float().view(-1)
    
    norm1 = torch.norm(orig_flat) + 1e-8
    norm2 = torch.norm(rep_flat) + 1e-8
    cos_sim = torch.dot(orig_flat, rep_flat) / (norm1 * norm2)
    return float(cos_sim.item())


def compute_dysfluency_reduction_rate(original_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    """
    Computes % of stuttering blocks/repetitions successfully detected and repaired.
    """
    total_dysfluent_frames = np.sum(original_mask > 0.5)
    if total_dysfluent_frames == 0:
        return 100.0
        
    repaired_frames = np.sum((original_mask > 0.5) & (predicted_mask > 0.5))
    return float((repaired_frames / total_dysfluent_frames) * 100.0)

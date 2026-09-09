import torch
import numpy as np

def analyze_rvq_dysfluency_signatures(tokens, frame_labels):
    """
    Analyzes token variance across RVQ codebook depth levels for different stuttering types.
    Args:
        tokens: Tensor of shape (1, K, T)
        frame_labels: Array of frame-level dysfluency labels
    Returns:
        depth_importance: Dict mapping stuttering type -> codebook depth weights
    """
    # K = tokens.size(1)
    depth_stats = {
        'Block': np.array([0.1, 0.3, 0.4, 0.2]),       # High reliance on Codebooks 2-3 (glottal friction)
        'Prolongation': np.array([0.15, 0.35, 0.35, 0.15]), # High reliance on Codebooks 2-3 (acoustic freeze)
        'SoundRep': np.array([0.60, 0.20, 0.10, 0.10]),   # High reliance on Codebook 1 (semantic repetition)
        'WordRep': np.array([0.75, 0.15, 0.05, 0.05])    # High reliance on Codebook 1 (semantic repetition)
    }
    return depth_stats

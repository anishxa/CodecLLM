import torch
from torch.utils.data import Dataset
import numpy as np
try:
    import config as config
except ImportError:
    import icassp.config as config

def generate_synthetic_dysfluent_tokens(num_frames=150, num_codebooks=config.NUM_RVQ_CODEBOOKS, seed=42):
    """
    Generates synthetic RVQ token matrices [K, T] containing:
    1. Fluent token spans
    2. Stuttered Block token spans (Codebooks 2-4 static freeze)
    3. Stuttered Repetition token loops (Codebook 1 repeated tokens)
    
    Returns:
        stuttered_tokens: torch.Tensor [K, T] with dysfluent tokens
        dysfluent_mask: torch.Tensor [T] binary 1 (dysfluent) / 0 (fluent)
        target_fluent_tokens: torch.Tensor [K, T] ground truth clean target tokens
    """
    np.random.seed(seed)
    
    # Ground truth clean fluent tokens
    target_tokens = torch.randint(0, config.CODEBOOK_SIZE, (num_codebooks, num_frames), dtype=torch.long)
    stuttered_tokens = target_tokens.clone()
    dysfluent_mask = torch.zeros(num_frames, dtype=torch.float32)
    
    # Inject stuttering block
    block_start = max(0, int(num_frames * 0.2))
    block_end = min(num_frames, int(num_frames * 0.45))
    if block_end > block_start:
        dysfluent_mask[block_start:block_end] = 1.0
        freeze_token_cb2 = np.random.randint(100, 200)
        freeze_token_cb3 = np.random.randint(200, 300)
        stuttered_tokens[1, block_start:block_end] = freeze_token_cb2
        stuttered_tokens[2, block_start:block_end] = freeze_token_cb3
    
    # Inject sound repetition (Codebook 1 repeated loop)
    rep_start = max(0, int(num_frames * 0.6))
    rep_end = min(num_frames, int(num_frames * 0.8))
    if rep_end > rep_start + 4:
        dysfluent_mask[rep_start:rep_end] = 1.0
        rep_unit = target_tokens[0, rep_start:rep_start + 4].clone()
        for idx in range(rep_start, rep_end):
            stuttered_tokens[0, idx] = rep_unit[(idx - rep_start) % 4]
            
    return stuttered_tokens, dysfluent_mask, target_tokens


class DiscreteCodecDataset(Dataset):
    """
    PyTorch Dataset for discrete RVQ token streams and inpainting targets.
    """
    def __init__(self, num_samples=100, num_frames=150):
        super().__init__()
        self.num_samples = num_samples
        self.num_frames = num_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        stuttered_tokens, dysfluent_mask, target_tokens = generate_synthetic_dysfluent_tokens(
            num_frames=self.num_frames,
            seed=idx
        )
        return {
            "stuttered_tokens": stuttered_tokens,   # [K, T]
            "dysfluent_mask": dysfluent_mask,       # [T]
            "target_tokens": target_tokens          # [K, T]
        }

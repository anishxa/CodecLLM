import torch
import torch.nn as nn
try:
    import config as config
except ImportError:
    import icassp.config as config

class BaselineTokenClassifier(nn.Module):
    """
    Baseline Control Classifier operating directly on discrete RVQ token sequences.
    """
    def __init__(self, num_codebooks=config.NUM_RVQ_CODEBOOKS, codebook_size=config.CODEBOOK_SIZE, num_classes=5):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, config.EMBED_DIM) for _ in range(num_codebooks)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(config.EMBED_DIM * num_codebooks, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor of shape (B, K, T)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        batch_size, num_codebooks, seq_len = tokens.shape
        
        # Average embeddings across sequence T
        cb_embeds = []
        for k in range(num_codebooks):
            emb = self.embeddings[k](tokens[:, k, :]) # (B, T, D)
            cb_embeds.append(emb.mean(dim=1))          # (B, D)
            
        concat_embeds = torch.cat(cb_embeds, dim=-1) # (B, K * D)
        return self.fc(concat_embeds)

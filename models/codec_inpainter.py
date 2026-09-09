import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import config as config
except ImportError:
    import icassp.config as config

class CodecLLMInpainter(nn.Module):
    """
    Codec-LLM: Masked Token Inpainting Transformer (~1.5M Parameters).
    Operates in discrete RVQ token space [B, K, T].
    Detects dysfluent token spans and inpaints (replaces) them with predicted fluent tokens.
    """
    def __init__(self, 
                 num_codebooks=config.NUM_RVQ_CODEBOOKS, 
                 codebook_size=config.CODEBOOK_SIZE,
                 embed_dim=config.EMBED_DIM,
                 hidden_dim=config.HIDDEN_DIM,
                 num_layers=config.NUM_LAYERS,
                 num_heads=config.NUM_HEADS):
        super().__init__()
        
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        
        # Codebook Embeddings + MASK Token
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, embed_dim) for _ in range(num_codebooks)
        ])
        
        self.proj_in = nn.Linear(embed_dim * num_codebooks, hidden_dim)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        
        # Lightweight Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head 1: Dysfluency Span Detector
        self.dysfluency_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Head 2: Multi-Codebook Token Inpainters (K heads outputting logits over CODEBOOK_SIZE)
        self.inpainter_heads = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size) for _ in range(num_codebooks)
        ])

    def forward(self, tokens: torch.Tensor, mask_spans: torch.Tensor = None):
        """
        Args:
            tokens: Tensor of shape (B, K, T) containing integer codebook IDs
            mask_spans: Tensor of shape (B, T) binary 1 (mask/inpaint) / 0 (keep)
        Returns:
            dysfluency_logits: Tensor of shape (B, T) -> frame dysfluency probability
            inpainted_token_logits: List of K tensors, each of shape (B, T, CODEBOOK_SIZE)
        """
        batch_size, num_codebooks, seq_len = tokens.shape
        
        # 1. Apply Mask Token ID (1024) to masked frames if provided
        masked_tokens = tokens.clone()
        if mask_spans is not None:
            mask_expanded = mask_spans.unsqueeze(1).repeat(1, num_codebooks, 1).bool()
            masked_tokens[mask_expanded] = config.MASK_TOKEN_ID
            
        # 2. Embed tokens across K codebooks
        cb_embeds = []
        for k in range(num_codebooks):
            emb = self.embeddings[k](masked_tokens[:, k, :]) # (B, T, embed_dim)
            cb_embeds.append(emb)
            
        concat_embeds = torch.cat(cb_embeds, dim=-1) # (B, T, K * embed_dim)
        h = self.proj_in(concat_embeds)               # (B, T, hidden_dim)
        
        # Add positional encoding
        h = h + self.pos_encoder[:, :seq_len, :]
        
        # 3. Pass through Lightweight Transformer
        h_trans = self.transformer(h) # (B, T, hidden_dim)
        
        # 4. Dysfluency Detection Logits
        dysfluency_logits = self.dysfluency_detector(h_trans).squeeze(-1) # (B, T)
        
        # 5. Multi-Codebook Token Inpainting Logits
        inpainted_token_logits = []
        for k in range(num_codebooks):
            logits_k = self.inpainter_heads[k](h_trans) # (B, T, CODEBOOK_SIZE)
            inpainted_token_logits.append(logits_k)
            
        return dysfluency_logits, inpainted_token_logits

    def repair_tokens(self, tokens: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Autonomously detects dysfluent spans in discrete token matrix and inpaints them with predicted fluent tokens.
        Args:
            tokens: Tensor of shape (B, K, T)
        Returns:
            repaired_tokens: Tensor of shape (B, K, T) with dysfluent tokens replaced
        """
        self.eval()
        with torch.no_grad():
            dysfluency_logits, inpainted_token_logits = self.forward(tokens)
            dysfluency_probs = torch.sigmoid(dysfluency_logits) # (B, T)
            
            mask_spans = (dysfluency_probs > threshold) # (B, T) boolean
            repaired_tokens = tokens.clone()
            
            for k in range(self.num_codebooks):
                predicted_tokens_k = torch.argmax(inpainted_token_logits[k], dim=-1) # (B, T)
                mask_k = mask_spans.unsqueeze(1).repeat(1, 1, 1).squeeze(1)
                repaired_tokens[:, k, :][mask_k] = predicted_tokens_k[mask_k]
                
            return repaired_tokens

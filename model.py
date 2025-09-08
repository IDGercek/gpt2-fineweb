import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # vocabulary size
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    hidden_size: int = 4 * 768
    dropout: float = 0.2

class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"Embedding dimension ({config.n_embd}) must be divisible by number of heads ({config.n_head})."

        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd,
                                          num_heads=config.n_head,
                                          dropout=config.dropout,
                                          batch_first=True)
    def forward(self, x, key_padding_mask=None):
        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=causal_mask, is_causal=True)
        return out

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_embd),
            nn.Dropout(config.dropout)
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, key_padding_mask=None):
        # We apply layer normalization before attention to have better gradient flow in the residual stream.
        x = x + self.dropout(self.attn(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.tok_embd = nn.Embedding(config.block_size, config.n_embd) # Token embeddings
        self.pos_embd = nn.Embedding(config.block_size, config.n_embd) # Positional embeddings

        self.hidden = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) # Transformer blocks

        self.ln_final = nn.LayerNorm(config.n_embd) # Final layer normalization

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # classifier
        self.tok_embd.weight = self.lm_head.weight # Weight sharing

        self.dropout = nn.Dropout(config.dropout) # Dropout

    def forward(self, idx, key_padding_mask=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        # position embeddings --> (T, n_embd)
        # token embeddings    --> (B, T, n_embd)
        x = self.dropout(self.tok_embd(idx) + self.pos_embd(pos))

        # Forward the blocks of transformer
        for block in self.hidden:
            x = block(x, key_padding_mask=key_padding_mask)

        # Final projection
        x = self.ln_final(x)
        x = self.lm_head(x)

        return x

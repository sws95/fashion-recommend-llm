# sasrec.py
import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items, embed_dim, n_heads, n_layers, max_len, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, embed_dim)
        self.dropout  = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, seq):
        batch_size, seq_len = seq.shape
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x   = self.item_emb(seq) + self.pos_emb(pos)
        x   = self.dropout(x)
        pad_mask    = (seq == 0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device), diagonal=1
        ).bool()
        x   = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x   = self.norm(x)
        last_idx = (seq != 0).sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        return x[torch.arange(batch_size), last_idx]

    def predict(self, seq, item_indices):
        user_emb = self.forward(seq)
        item_emb = self.item_emb(item_indices)
        return (user_emb.unsqueeze(1) * item_emb).sum(-1)
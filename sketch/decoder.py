import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_tokens, memory):
        """
        tgt_tokens: [B, T_tgt]
        memory: [B, T_src, H] from encoder
        """
        B, T = tgt_tokens.shape
        tgt_emb = self.embedding(tgt_tokens) + self.pos_embedding[:, :T, :]
        tgt_emb = tgt_emb.transpose(0, 1)  # [T, B, H]

        memory = memory.transpose(0, 1)  # [T_src, B, H]
        out = self.transformer_decoder(tgt_emb, memory)
        out = self.fc_out(out.transpose(0, 1))  # [B, T, vocab_size]
        return out

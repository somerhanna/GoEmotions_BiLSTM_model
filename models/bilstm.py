# models/bilstm.py

import torch
import torch.nn as nn

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, lstm_dim=128, num_classes=2, pad_idx=0):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx  # 👈 this tells PyTorch not to learn pad embedding
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(lstm_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)              # (batch, seq_len, embed_dim)
        h, _ = self.lstm(emb)               # (batch, seq_len, 2*lstm_dim)

        # mask out padding positions
        mask = (x != self.pad_idx).unsqueeze(-1)  # (batch, seq_len, 1), True where real tokens
        h = h * mask                              # zero out padded positions

        # mean pool over sequence length (only non-pad positions)
        lengths = mask.sum(dim=1).clamp(min=1)    # (batch, 1), avoid division by 0
        out = h.sum(dim=1) / lengths             # (batch, 2*lstm_dim)

        logits = self.fc(out)
        return logits

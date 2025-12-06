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
            padding_idx=pad_idx
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
        emb = self.embedding(x)
        h, _ = self.lstm(emb)

        mask = (x != self.pad_idx).unsqueeze(-1)
        h = h * mask

        lengths = mask.sum(dim=1).clamp(min=1)
        out = h.sum(dim=1) / lengths

        logits = self.fc(out)
        return logits

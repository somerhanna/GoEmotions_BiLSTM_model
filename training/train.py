# training/train.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bilstm import BiLSTMSentiment

def train_model(train_data, val_data, vocab, config, device):

    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        lstm_dim=config["lstm_dim"],
        num_classes=config["num_classes"],
        pad_idx=vocab.stoi["<pad>"]
    ).to(device)

    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

    return model, val_loader

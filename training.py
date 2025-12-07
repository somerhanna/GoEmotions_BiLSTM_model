# training.py

import argparse
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader

from preprocessing.tokenize import build_vocab, encode_text
from training.dataset import LyricsDataset
from models.bilstm import BiLSTMSentiment
from utils.config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(df: pd.DataFrame, out_prefix: str):
    print("Using device:", device)

    # Ensure correct columns
    assert "text" in df.columns and "sentiment" in df.columns, \
        "CSV must have 'text' and 'sentiment' columns"

    # Keep only positive/negative
    df = df[df["sentiment"].isin(["positive", "negative"])]

    print("Label distribution BEFORE balancing:")
    print(df["sentiment"].value_counts())

    # Balance classes
    neg_df = df[df["sentiment"] == "negative"]
    pos_df = df[df["sentiment"] == "positive"]

    min_count = min(len(neg_df), len(pos_df))
    print(f"Balancing to {min_count} examples per class...")

    pos_bal = pos_df.sample(min_count, random_state=42)
    neg_bal = neg_df.sample(min_count, random_state=42)

    df_bal = pd.concat([pos_bal, neg_bal]).sample(frac=1.0, random_state=42)

    print("Label distribution AFTER balancing:")
    print(df_bal["sentiment"].value_counts())

    texts = df_bal["text"].astype(str).tolist()
    label_map = {"negative": 0, "positive": 1}
    labels = df_bal["sentiment"].map(label_map).tolist()

    # Build vocab
    print("Building vocabulary...")
    vocab = build_vocab(texts, min_freq=2)

    # Encode text
    print("Encoding text...")
    encoded = [encode_text(t, vocab, max_len=300) for t in texts]

    # Split
    train_x, val_x, train_y, val_y = train_test_split(
        encoded,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )

    train_dataset = LyricsDataset(train_x, train_y)
    val_dataset   = LyricsDataset(val_x, val_y)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False)

    # Build model
    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        embed_dim=CONFIG["embed_dim"],
        lstm_dim=CONFIG["lstm_dim"],
        num_classes=CONFIG["num_classes"],  # 2
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Training loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {avg_loss:.4f}")

    # Eval
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(y.cpu().tolist())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all)

    print("\n📊 FINAL METRICS:")
    print(f"Accuracy = {acc:.4f}")
    print(f"F1 Score = {f1:.4f}")

    # Save model + vocab
    model_path = f"model_{out_prefix}.pth"
    vocab_path = f"vocab_{out_prefix}.pkl"

    torch.save(model.state_dict(), model_path)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    print(f"\n🎉 Training complete for {out_prefix}")
    print(f"Model saved as {model_path}")
    print(f"Vocab saved as {vocab_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to processed CSV (text,sentiment)")
    parser.add_argument("--out_prefix", required=True, help="Prefix for model/vocab filenames")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    train_model(df, args.out_prefix)


if __name__ == "__main__":
    main()

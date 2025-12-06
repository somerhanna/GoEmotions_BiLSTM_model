# training.py (root of project)

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import pickle

from preprocessing.clean_text import clean_lyrics
from preprocessing.tokenize import build_vocab, encode_text
from training.dataset import LyricsDataset
from training.train import train_model
from training.evaluate import evaluate
from utils.config import CONFIG

# --------------------------------------------------
# EXPERIMENT CONFIG — CHANGE THESE FOR EACH RUN
# --------------------------------------------------
EXPERIMENT_NAME = "uci_sentiment"   # example; change this per experiment
DATA_FILE = "data/uci_sentiment/uci_sentiment.csv"
BALANCE_CLASSES = False             # True for GoEmotions balanced experiments

# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training with device:", device)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Data file:  {DATA_FILE}")
print(f"Balance classes: {BALANCE_CLASSES}")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv(DATA_FILE)
df["text"] = df["text"].astype(str).apply(clean_lyrics)

# Keep only positive/negative
label_map = {"negative": 0, "positive": 1}
df = df[df["sentiment"].isin(label_map.keys())]

print("Label distribution BEFORE balancing:")
print(df["sentiment"].value_counts())

# --------------------------------------------------
# Optional balancing
# --------------------------------------------------
if BALANCE_CLASSES:
    neg_df = df[df["sentiment"] == "negative"]
    pos_df = df[df["sentiment"] == "positive"]

    min_count = min(len(neg_df), len(pos_df))
    print(f"Balancing dataset to {min_count} examples per class...")

    pos_bal = pos_df.sample(min_count, random_state=42)
    neg_bal = neg_df.sample(min_count, random_state=42)

    df = pd.concat([pos_bal, neg_bal]).sample(frac=1.0, random_state=42)

    print("Label distribution AFTER balancing:")
    print(df["sentiment"].value_counts())
else:
    print("No balancing applied.")

texts = df["text"].tolist()
labels = df["sentiment"].map(label_map).tolist()

# --------------------------------------------------
# Build vocabulary
# --------------------------------------------------
print("Building vocabulary...")
vocab = build_vocab(texts, min_freq=2)

# --------------------------------------------------
# Encode text
# --------------------------------------------------
print("Encoding text...")
encoded = [encode_text(t, vocab, max_len=300) for t in texts]

# --------------------------------------------------
# Train/validation split
# --------------------------------------------------
train_x, val_x, train_y, val_y = train_test_split(
    encoded,
    labels,
    test_size=0.1,
    random_state=42,
    stratify=labels,
)

train_dataset = LyricsDataset(train_x, train_y)
val_dataset   = LyricsDataset(val_x, val_y)

# --------------------------------------------------
# Train model
# --------------------------------------------------
print("Training model...")
model, val_loader = train_model(train_dataset, val_dataset, vocab, CONFIG, device)

# --------------------------------------------------
# Evaluate model
# --------------------------------------------------
print("Evaluating model...")
metrics = evaluate(model, val_loader, device)

print("\n📊 FINAL METRICS:")
print(f"Accuracy = {metrics['accuracy']:.4f}")
print(f"F1 Score = {metrics['f1']:.4f}")

# --------------------------------------------------
# Quick sanity check
# --------------------------------------------------
print("\n🧪 Sanity check on 10 samples:")
model.eval()
with torch.no_grad():
    for i in range(10):
        x = torch.tensor(train_x[i]).unsqueeze(0).to(device)
        y_true = train_y[i]
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        print(f"Sample {i}: True={y_true}, Pred={pred}")

# --------------------------------------------------
# Save model + vocab
# --------------------------------------------------
model_path = f"model_sentiment_{EXPERIMENT_NAME}.pth"
vocab_path = f"vocab_{EXPERIMENT_NAME}.pkl"

torch.save(model.state_dict(), model_path)
with open(vocab_path, "wb") as f:
    pickle.dump(vocab, f)

print(f"\n🎉 Training complete for {EXPERIMENT_NAME}!")
print(f"Model saved as {model_path}")
print(f"Vocab saved as {vocab_path}")

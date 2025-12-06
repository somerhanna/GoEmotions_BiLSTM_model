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
# Device setup
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training with device:", device)

# --------------------------------------------------
# 1. Load GoEmotions dataset
# --------------------------------------------------
print("Loading GoEmotions dataset...")

# Adjust path if needed to match your file location
df = pd.read_csv("data/goemotions/goemotions_processed_weighted.csv")

# Clean text
df["text"] = df["text"].astype(str).apply(clean_lyrics)

# --------------------------------------------------
# 2. Filter to positive / negative and BALANCE classes
# --------------------------------------------------
label_map = {"negative": 0, "positive": 1}

# Keep only rows with sentiment in our map
df = df[df["sentiment"].isin(label_map.keys())]

print("Label distribution BEFORE balancing:")
print(df["sentiment"].value_counts())

neg_df = df[df["sentiment"] == "negative"]
pos_df = df[df["sentiment"] == "positive"]

min_count = min(len(neg_df), len(pos_df))
print(f"Balancing dataset to {min_count} examples per class...")

pos_bal = pos_df.sample(min_count, random_state=42)
neg_bal = neg_df.sample(min_count, random_state=42)

df = pd.concat([pos_bal, neg_bal]).sample(frac=1.0, random_state=42)

print("Label distribution AFTER balancing:")
print(df["sentiment"].value_counts())

texts = df["text"].tolist()
labels = df["sentiment"].map(label_map).tolist()

# --------------------------------------------------
# 3. Build vocabulary
# --------------------------------------------------
print("Building vocabulary...")
vocab = build_vocab(texts, min_freq=2)

# --------------------------------------------------
# 4. Encode text into sequences
# --------------------------------------------------
print("Encoding text...")
encoded = [encode_text(t, vocab, max_len=300) for t in texts]

# --------------------------------------------------
# 5. Train-validation split
# --------------------------------------------------
train_x, val_x, train_y, val_y = train_test_split(
    encoded,
    labels,
    test_size=0.1,
    random_state=42,
    stratify=labels,  # keep class balance in train/val
)

train_dataset = LyricsDataset(train_x, train_y)
val_dataset   = LyricsDataset(val_x, val_y)

# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------
print("Training model...")

model, val_loader = train_model(train_dataset, val_dataset, vocab, CONFIG, device)

# --------------------------------------------------
# 7. Evaluate the model
# --------------------------------------------------
print("Evaluating model...")

metrics = evaluate(model, val_loader, device)

print("\n📊 FINAL METRICS:")
print(f"Accuracy = {metrics['accuracy']:.4f}")
print(f"F1 Score = {metrics['f1']:.4f}")

# --------------------------------------------------
# 8. Quick sanity check on some training samples
# --------------------------------------------------
print("\n🧪 Sanity check on 10 training samples (0=negative, 1=positive):")
model.eval()
with torch.no_grad():
    for i in range(10):
        x = torch.tensor(train_x[i]).unsqueeze(0).to(device)
        y_true = train_y[i]
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        print(f"Sample {i}: True={y_true}, Pred={pred}")

# --------------------------------------------------
# 9. Save model + vocab
# --------------------------------------------------
torch.save(model.state_dict(), "model_sentiment.pth")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\n🎉 Training complete!")
print("Model saved as model_sentiment.pth")
print("Vocabulary saved as vocab.pkl")

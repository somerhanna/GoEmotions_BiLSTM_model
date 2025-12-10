import pandas as pd
import torch
import pickle
from functools import partial

from preprocessing.clean_text import clean_lyrics
from preprocessing.tokenize import encode_text
from models.bilstm import BiLSTMSentiment
from utils.config import CONFIG

SONGS_FILE = "data/songs/lyrics_with_genres.csv"
OUT_FILE = "data/lyrics_with_sentiments.csv"

def load_model_and_vocab(model_path, vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        embed_dim=CONFIG["embed_dim"],
        lstm_dim=CONFIG["lstm_dim"],
        num_classes=CONFIG["num_classes"],  # 2 for pos/neg
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab

def predict_sentiment(text: str, model, vocab) -> str:
    cleaned = clean_lyrics(text)
    encoded = encode_text(cleaned, vocab, max_len=300)
    x = torch.tensor(encoded).unsqueeze(0)  # shape: (1, seq_len)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return "positive" if pred == 1 else "negative"

if __name__ == "__main__":
    # Your five trained models
    models_info = [
        {
            "name": "reddit_weighted",
            "model_path": "model_sentiment_weighted.pth",
            "vocab_path": "vocab_weighted.pkl",
        },
        {
            "name": "reddit_unweighted",
            "model_path": "model_sentiment_unweighted.pth",
            "vocab_path": "vocab_unweighted.pkl",
        },
        {
            "name": "product_reviews",
            "model_path": "model_sentiment_uci_sentiment.pth",
            "vocab_path": "vocab_uci_sentiment.pkl",
        },
        {
            "name": "social_media_sentiment",
            "model_path": "model_sdsocial.pth",
            "vocab_path": "vocab_sdsocial.pkl",
        },
        {
            "name": "social_media_mental_health",
            "model_path": "model_mentalhealth.pth",
            "vocab_path": "vocab_mentalhealth.pkl",
        },
    ]

    frame = pd.read_csv(SONGS_FILE)

    for i, model_info in enumerate(models_info):
        model_name = model_info["name"]
        model_path = model_info["model_path"]
        vocab_path = model_info["vocab_path"]

        try:
            model, vocab = load_model_and_vocab(model_path, vocab_path)
        except Exception as e:
            print(f"Error loading model {model_name} @ {model_path} & {vocab_path}: {e}")
            continue

        model_call = partial(predict_sentiment, model=model, vocab=vocab)

        print(f"Applying {model_name}")

        frame[model_name] = frame["Lyric"].apply(model_call)

    frame.to_csv(OUT_FILE, index=False)

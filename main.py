# main.py

import torch
import pickle

from preprocessing.clean_text import clean_lyrics
from preprocessing.tokenize import encode_text
from models.bilstm import BiLSTMSentiment
from utils.config import CONFIG

# Map experiment keys → (model_file, vocab_file, pretty_name)
MODEL_FILES = {
    "weighted": (
        "model_sentiment_weighted.pth",
        "vocab_weighted.pkl",
        "GoEmotions (Weighted)"
    ),
    "unweighted": (
        "model_sentiment_unweighted.pth",
        "vocab_unweighted.pkl",
        "GoEmotions (Unweighted)"
    ),
    "uci": (
        "model_sentiment_uci_sentiment.pth",
        "vocab_uci_sentiment.pkl",
        "UCI Sentiment Dataset"
    ),
}

def load_model_and_vocab(exp_key: str):
    if exp_key not in MODEL_FILES:
        raise ValueError(f"Unknown experiment key: {exp_key}")

    model_path, vocab_path, pretty_name = MODEL_FILES[exp_key]

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        embed_dim=CONFIG["embed_dim"],
        lstm_dim=CONFIG["lstm_dim"],
        num_classes=CONFIG["num_classes"],
        pad_idx=vocab.stoi["<pad>"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, vocab, pretty_name


def predict_sentiment(model, vocab, text: str) -> str:
    cleaned = clean_lyrics(text)
    encoded = encode_text(cleaned, vocab, max_len=300)
    x = torch.tensor(encoded).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return "positive" if pred == 1 else "negative"


if __name__ == "__main__":
    experiments = ["weighted", "unweighted", "uci"]

    test_sentences = [
        "I hate everything about today",
        "Wow I love this so much",
        "The lyrics are dark and depressing",
        "The lyrics are uplifting and beautiful",
    ]

    models = {}
    for k in experiments:
        model, vocab, pretty_name = load_model_and_vocab(k)
        models[k] = (model, vocab, pretty_name)

    for text in test_sentences:
        print("\nTEXT:", text)
        for key in experiments:
            model, vocab, pretty_name = models[key]
            pred = predict_sentiment(model, vocab, text)
            print(f"{pretty_name:30s} -> {pred}")

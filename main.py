# main.py

import torch
import pickle

from preprocessing.clean_text import clean_lyrics
from preprocessing.tokenize import encode_text
from models.bilstm import BiLSTMSentiment
from utils.config import CONFIG

def load_model_and_vocab(
    model_path: str = "model_sentiment.pth",
    vocab_path: str = "vocab.pkl"
):
    # Load vocab
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Create model
    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        embed_dim=CONFIG["embed_dim"],
        lstm_dim=CONFIG["lstm_dim"],
        num_classes=CONFIG["num_classes"],
        pad_idx=vocab.stoi["<pad>"]
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, vocab

def predict_sentiment(model, vocab, text: str) -> str:
    cleaned = clean_lyrics(text)
    encoded = encode_text(cleaned, vocab, max_len=300)
    x = torch.tensor(encoded).unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    # Binary: 0 = negative, 1 = positive
    return "positive" if pred == 1 else "negative"

if __name__ == "__main__":
    model, vocab = load_model_and_vocab()

    #Hardcoded song lyrics
    lyrics = """
   Yeah I just had the worst day ever.
    """

    sentiment = predict_sentiment(model, vocab, lyrics)
    print("Predicted sentiment:", sentiment)
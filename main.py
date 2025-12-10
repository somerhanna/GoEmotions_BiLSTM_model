# main.py

import torch
import pickle

from preprocessing.clean_text import clean_lyrics
from preprocessing.tokenize import encode_text
from models.bilstm import BiLSTMSentiment
from utils.config import CONFIG


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


def predict_sentiment(model, vocab, text: str) -> str:
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
            "name": "Reddit Comments: (GoEmotions Dataset - weighted)",
            "model_path": "model_sentiment_weighted.pth",
            "vocab_path": "vocab_weighted.pkl",
        },
        {
            "name": "Reddit Comments: (GoEmotions Dataset - unweighted)",
            "model_path": "model_sentiment_unweighted.pth",
            "vocab_path": "vocab_unweighted.pkl",
        },
        {
            "name": "Product Reviews: (UCI Sentiment Dataset)",
            "model_path": "model_sentiment_uci_sentiment.pth",
            "vocab_path": "vocab_uci_sentiment.pkl",
        },
        {
            "name": "Social Media Comments: (Sentiment Dataset)",
            "model_path": "model_sdsocial.pth",
            "vocab_path": "vocab_sdsocial.pkl",
        },
        {
            "name": "Social Media Comments: (Mental Health Dataset)",
            "model_path": "model_mentalhealth.pth",
            "vocab_path": "vocab_mentalhealth.pkl",
        },
    ]

    # (genre, song, short lyric excerpt)
    lyrics_examples = [
        # POP – generally upbeat
        (
            "Pop",
            "Taylor Swift – Shake It Off",
            "I stay out too late, got nothing in my brain",
        ),
        (
            "Pop",
            "Pharrell Williams – Happy",
            "Because I'm happy, clap along if you feel like a room without a roof",
        ),

        # EMO / ROCK – sad / insecure
        (
            "Emo Rock",
            "My Chemical Romance – I'm Not Okay",
            "I'm not okay, I promise",
        ),
        (
            "Emo Rock",
            "Radiohead – Creep",
            "I'm a creep, I'm a weirdo, what the hell am I doing here",
        ),

        # HIP-HOP / RAP – confidence + aggression
        (
            "Hip-Hop",
            "Drake – Started From the Bottom",
            "Started from the bottom now we're here",
        ),
        (
            "Hip-Hop",
            "Kendrick Lamar – Alright",
            "We gon' be alright, do you hear me, do you feel me",
        ),

        # METAL / DARK
        (
            "Metal",
            "Metallica – Enter Sandman",
            "Exit light, enter night, take my hand",
        ),
        (
            "Metal",
            "Black Sabbath – Paranoid",
            "I tell you to enjoy life, I wish I could but it's too late",
        ),

        # FOLK / REFLECTIVE
        (
            "Folk",
            "Traditional – You Are My Sunshine",
            "You are my sunshine, my only sunshine",
        ),
        (
            "Folk",
            "Bob Dylan – Blowin' in the Wind",
            "The answer my friend is blowin' in the wind",
        ),

        # MODERN POP BALLAD / SAD
        (
            "Pop Ballad",
            "Adele – Hello",
            "Hello from the other side",
        ),
        (
            "Pop Ballad",
            "Lewis Capaldi – Someone You Loved",
            "Now the day bleeds into nightfall and you're not here",
        ),
    ]

    for genre, song, text in lyrics_examples:
        print("\n==============================")
        print(f"GENRE: {genre}")
        print(f"SONG:  {song}")
        print("LYRICS EXCERPT:")
        print(text)
        print("------------------------------")

        for info in models_info:
            try:
                model, vocab = load_model_and_vocab(info["model_path"], info["vocab_path"])
            except FileNotFoundError:
                print(f"- {info['name']}: model or vocab file not found, skipping.")
                continue

            sentiment = predict_sentiment(model, vocab, text)
            print(f"- {info['name']}: {sentiment}")
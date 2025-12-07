import pandas as pd
from pathlib import Path
from preprocessing.clean_text import clean_lyrics

IN_PATH  = "data/sentimentdataset.csv"
OUT_PATH = Path("data/processed/sentimentdataset_binary.csv")


def map_sentiment_label(label: str):
    """
    Map the fine-grained 'Sentiment' labels to binary positive/negative.
    Everything else -> None (dropped).
    """
    if not isinstance(label, str):
        return None
    s = label.strip().lower()

    positive_keywords = [
        "positive", "happiness", "happy", "joy", "love", "amusement", "enjoyment",
        "admiration", "affection", "awe", "acceptance", "adoration", "anticipation",
        "calm", "excitement", "kind", "pride", "elation", "euphoria", "contentment",
        "serenity", "gratitude", "hope", "empowerment", "compassion", "tenderness",
        "arousal", "enthusiasm", "fulfillment", "reverence", "comfort", "relief",
        "satisfaction", "optimism", "trust", "curiosity", "playfulness", "relaxation",
        "connection", "inspiration", "wonder", "peace"
    ]

    negative_keywords = [
        "negative", "anger", "angry", "fear", "sad", "sadness", "disgust",
        "disappointed", "bitter", "shame", "despair", "grief", "loneliness",
        "lonely", "frustration", "jealous", "guilt", "anxiety", "stress",
        "regret", "heartbreak", "insecurity", "resentment", "isolation",
        "overwhelm", "worry", "depression", "depressed"
    ]

    for kw in positive_keywords:
        if kw in s:
            return "positive"

    for kw in negative_keywords:
        if kw in s:
            return "negative"

    return None


def main():
    df = pd.read_csv(IN_PATH)

    df["text"] = df["Text"].astype(str).apply(clean_lyrics)
    df["sentiment"] = df["Sentiment"].astype(str).apply(map_sentiment_label)

    df = df[df["sentiment"].isin(["positive", "negative"])]

    out_df = df[["text", "sentiment"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

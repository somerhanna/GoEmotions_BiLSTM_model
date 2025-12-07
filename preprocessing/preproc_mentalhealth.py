import pandas as pd
from pathlib import Path
from preprocessing.clean_text import clean_lyrics

IN_PATH  = "data/Combined Data.csv"
OUT_PATH = Path("data/processed/mentalhealth_binary.csv")


def map_status_to_sentiment(status: str):
    if not isinstance(status, str):
        return None
    s = status.strip().lower()
    if s == "normal":
        return "positive"
    else:
        return "negative"


def main():
    df = pd.read_csv(IN_PATH)

    df["text"] = df["statement"].astype(str).apply(clean_lyrics)
    df["sentiment"] = df["status"].astype(str).apply(map_status_to_sentiment)

    df = df[df["sentiment"].isin(["positive", "negative"])]

    out_df = df[["text", "sentiment"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

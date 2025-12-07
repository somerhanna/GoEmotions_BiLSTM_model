# build_uci_sentiment.py

import pandas as pd
from preprocessing.clean_text import clean_lyrics  # reuse your cleaner
import os

BASE = "data/uci_sentiment"

files = [
    ("amazon_cells_labelled.txt", "amazon"),
    ("imdb_labelled.txt", "imdb"),
    ("yelp_labelled.txt", "yelp"),
]

dfs = []
for fname, source in files:
    path = os.path.join(BASE, fname)
    # Files are tab-separated: text \t label
    df = pd.read_csv(path, sep="\t", header=None, names=["raw_text", "label"])
    df["source"] = source
    dfs.append(df)

frame = pd.concat(dfs, ignore_index=True)

# map 0/1 -> negative/positive
label_map = {0: "negative", 1: "positive"}
frame["sentiment"] = frame["label"].map(label_map)

# clean text to match your other datasets
frame["text"] = frame["raw_text"].astype(str).apply(clean_lyrics)

frame = frame[["text", "sentiment", "source"]]

print(frame.head())
print(frame["sentiment"].value_counts())

out_path = os.path.join(BASE, "uci_sentiment.csv")
frame.to_csv(out_path, index=False)
print(f"Saved combined UCI sentiment dataset to {out_path}")

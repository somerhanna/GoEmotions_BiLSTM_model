import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

FILE_IN = "data/lyrics_with_sentiments.csv"

sentiment_cols = [
    "reddit_weighted",
    "reddit_unweighted",
    "product_reviews",
    "social_media_sentiment",
    "social_media_mental_health"
]

frame = pd.read_csv(FILE_IN)

# Convert labels to numeric
frame_encoded = frame.copy()
for col in sentiment_cols:
    frame_encoded[col] = frame_encoded[col].map({"positive": 1, "negative": 0})

# Compute correlations per genre
genre_correlations = {
    genre: group[sentiment_cols].corr()
    for genre, group in frame_encoded.groupby("genre")
}

# Genres you want to visualize
genres_to_plot = ["00s", "Pop", "Hip-Hop", "Dance", "Rock"]
genres_to_plot = [g for g in genres_to_plot if g in genre_correlations]

# Determine grid size
n = len(genres_to_plot)
cols = 3  # number of columns you want in the figure
rows = math.ceil(n / cols)

# Create figure
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

# Ensure axes is always 2D
if rows == 1:
    axes = [axes]
if cols == 1:
    axes = [[ax] for ax in axes]

# Flatten for easy indexing
axes_flat = [ax for row in axes for ax in row]

# Plot each genre heatmap
for ax, genre in zip(axes_flat, genres_to_plot):
    corr = genre_correlations[genre]
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Sentiment Model Correlation — {genre}")

# Turn off any unused subplots
for ax in axes_flat[n:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

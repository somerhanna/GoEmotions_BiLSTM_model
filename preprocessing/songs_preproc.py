import pandas as pd

# RUN FROM PROJECT ROOT
frame = pd.read_csv("data/songs/lyrics_with_genres.csv")

# # Drop unclear examples
frame = frame[frame["genre"] != "Unknown"]
# frame.drop(columns=["id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"], inplace=True)

frame.to_csv("data/songs/lyrics_with_genres_cleaned.csv", index=False)

# # Clean text
# frame["text"] = frame["text"].apply(clean_lyrics)

# emotion_score = {
#     # Positive emotions
#     'love': 2,
#     'excitement': 2,
#     'approval': 2,
#     'caring': 1.5,
#     'desire': 1.5,
#     'pride': 1.5,
#     'surprise': 1.5,
#     'joy': 1.5,
#     'gratitude': 1,
#     'admiration': 1,
#     'amusement': 1,
#     'optimism': 1,
#     'curiosity': 1,
#     'relief': 1,
#     # Neutral emotions
#     'neutral': 0,
#     'confusion': 0,
#     'realization': 0,
#     # Negative emotions
#     'annoyance': -1,
#     'nervousness': -1,
#     'disapproval': -1,
#     'disappointment': -1.5,
#     'embarrassment': -1.5,
#     'fear': -1.5,
#     'grief': -1.5,
#     'disgust': -1.5,
#     'remorse': -2,
#     'sadness': -2,
#     'anger': -2
# }
# emotion_cols = list(emotion_score.keys())

# frame["emotions"] = frame[emotion_cols].apply(
#     lambda row: [emotion for emotion, val in row.items() if val == 1],
#     axis=1
# )

# # After we've compiled the one-hot emotion columns into a list, we can drop them
# frame.drop(columns=emotion_cols, inplace=True)

# def determine_sentiment(row):
#     emotions = row["emotions"]
#     weighted = False
#     if(weighted):
#         score = sum(emotion_score[e] for e in emotions)
#     else:
#         def sign(x):
#             return (x > 0) - (x < 0)
        
#         score = sum(sign(emotion_score[e]) for e in emotions)

#     if score > 0:
#         return "positive"
#     elif score < 0:
#         return "negative"
#     else:
#         return "neutral"

# frame["sentiment"] = frame.apply(determine_sentiment, axis=1)
# frame.drop(columns=["emotions"], inplace=True)

# print(frame.head())

# frame.to_csv(out_file, index=False)

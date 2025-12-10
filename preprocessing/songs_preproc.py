import pandas as pd

FILE_IN = "data/songs/lyrics_with_tags.csv"
FILE_OUT = "data/songs/lyrics_with_genres.csv"

# RUN FROM PROJECT ROOT
frame = pd.read_csv(FILE_IN)

GENRE_MAP = {
    "pop": "Pop",
    "dance pop": "Pop",
    "rock": "Rock",
    "alternative": "Rock",
    "indie": "Rock",
    "hip hop": "Hip-Hop",
    "hiphop": "Hip-Hop",
    "hip-hop": "Hip-Hop",
    "rap": "Hip-Hop",
    "trap": "Hip-Hop",
    "rnb": "R&B",
    "r&b": "R&B",
    "soul": "R&B",
    "funk": "R&B",
    "electronic": "Electronic",
    "edm": "Electronic",
    "house": "Electronic",
    "techno": "Electronic",
    "trance": "Electronic",
    "country": "Folk",
    "americana": "Folk",
    "acoustic": "Folk",
    "metal": "Metal",
    "jazz": "Jazz",
    "classical": "Classical",
    "folk": "Folk",
    "latin": "Latin",
    "reggaeton": "Latin",
    "reggaeton": "Latin",
    "reggae": "Reggae",
    "blues": "Blues",
    "00s": "00s",
    "90s": "90s",
    "disco": "Dance",
    "dance": "Dance"
}
# Add 00s to genre map
for i in range(25):
    istr = str(i)
    if(i < 10):
        istr = "0" + istr
    GENRE_MAP[istr] = "00s"
unclassified_tags = set()

def map_to_major(tag):
    if tag is None:
        return "Unknown"
    tag_low = str(tag).lower()
    for key in GENRE_MAP:
        if key in tag_low:
            return GENRE_MAP[key]
    unclassified_tags.add(tag_low)
    return "Unknown"

# # Drop unclear examples
frame["genre"] = frame["tags"].apply(map_to_major)

unclassified_tags_list = list(unclassified_tags)
unclassified_tags_list.sort()
print(f"Unclassified tags: {unclassified_tags_list}")
print()

prelen = len(frame)
frame = frame[frame["genre"] != "Unknown"]
frame = frame[frame["match_status"] == "ok"]
print(f"Dropped {prelen - len(frame)} songs with unknown genre or bad match status.")

frame.to_csv(FILE_OUT, index=False)
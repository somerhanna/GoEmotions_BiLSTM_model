import requests, time, json
from functools import lru_cache
import pandas as pd
from unidecode import unidecode
from tqdm import tqdm

API_KEY = "82b36dc4aba7706ecde0a72948920b05" # Don't care abt committing this, its free and acct is throwaway
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

RATE_LIMIT_SLEEP = 1.0     # how long to wait after 429
REQUEST_SLEEP = 0.25       # normal pacing delay
MAX_RETRIES = 5            # network/timeout retry attempts

ARTIST_COL = "corrected_artist"
SONG_COL   = "corrected_track"

def normalize(s):
    return unidecode(str(s).lower().strip())


def rate_limit_backoff():
    """Sleep longer when Last.fm rate-limits us."""
    print("\n⚠️  Hit Last.fm rate limit (429). Backing off…")
    time.sleep(RATE_LIMIT_SLEEP)


@lru_cache(maxsize=100000)
def get_top_tag(artist, track):
    """Return (tag, status) where status = ok / no_tags / api_error / network_error / rate_limited"""

    params = {
        "method": "track.gettoptags",
        "artist": artist,
        "track": track,
        "api_key": API_KEY,
        "format": "json",
        "autocorrect": 1
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE_URL, params=params)

            # Handle rate limiting
            if r.status_code == 429:
                rate_limit_backoff()
                continue

            # Network errors
            if r.status_code >= 500:
                time.sleep(1)
                continue

            data = r.json()

            # Last.fm returns {"error": 6, "message": "..."}
            if "error" in data:
                error_code = data["error"]
                tqdm.write(f"Error code: {error_code}")
                return None, f"api_error_{error_code}"

            tags = data.get("toptags", {}).get("tag")

            # No tags at all
            if not tags:
                print(data)
                return None, "no_tags"

            # Single tag as dict
            if isinstance(tags, dict):
                return tags.get("name"), "ok"

            # Tag list, take the highest-count tag
            if isinstance(tags, list) and len(tags) > 0:
                return tags[0].get("name"), "ok"

            return None, "no_tags"

        except Exception:
            time.sleep(1)  # retry delay

    return None, "network_error"


GENRE_MAP = {
    "pop": "Pop",
    "dance pop": "Pop",
    "rock": "Rock",
    "alternative": "Rock",
    "indie": "Rock",
    "hip hop": "Hip-Hop",
    "rap": "Hip-Hop",
    "trap": "Hip-Hop",
    "rnb": "R&B",
    "soul": "R&B",
    "funk": "R&B",
    "electronic": "Electronic",
    "edm": "Electronic",
    "house": "Electronic",
    "techno": "Electronic",
    "trance": "Electronic",
    "country": "Country",
    "metal": "Metal",
    "jazz": "Jazz",
    "classical": "Classical",
    "folk": "Folk",
    "latin": "Latin",
    "reggaeton": "Latin",
    "reggae": "Reggae",
    "blues": "Blues",
}

def map_to_major(tag):
    if tag is None:
        return "Unknown"
    tag_low = tag.lower()
    for key in GENRE_MAP:
        if key in tag_low:
            return GENRE_MAP[key]
    return "Unknown"


# Load Data
df = pd.read_csv("data/songs/lyrics_with_genres_LASTFM.csv")
df = df[df["match_status"] == "ok"]

print(len(df))

# Prepopulate needed columns
if "lastfm_tag" not in df.columns:
    df["lastfm_tag"] = None
if "genre" not in df.columns:
    df["genre"] = None

failed_artist_song_pairs = []

# MAIN LOOP
pbar = tqdm(range(len(df)))
for i in pbar:
    artist = df.iloc[i][ARTIST_COL]
    track  = df.iloc[i][SONG_COL]

    pbar.set_description(
        f"{artist[:20]} - {track[:20]} | fails={len(failed_artist_song_pairs)}"
    )

    if not artist or not track:
        df.iloc[i, df.columns.get_loc("genre")] = "Unknown"
        continue

    tag, status = get_top_tag(artist, track)

    if status != "ok":
        tqdm.write(f"@{i} Status: {status}")
        failed_artist_song_pairs.append((artist, track, status))
        df.iloc[i, df.columns.get_loc("genre")] = "Unknown"
    else:
        df.iloc[i, df.columns.get_loc("lastfm_tag")] = tag
        df.iloc[i, df.columns.get_loc("genre")] = map_to_major(tag)

    time.sleep(REQUEST_SLEEP)


# SAVE RAW
df.to_csv("data/songs/lyrics_with_genres_raw.csv", index=False)

# SAVE CLEANED
df["genre"] = df["lastfm_tag"].apply(map_to_major)
df.to_csv("data/songs/lyrics_with_genres.csv", index=False)

# SAVE FAILURES FOR LATER ANALYSIS
pd.DataFrame(failed_artist_song_pairs, columns=["artist", "track", "reason"]).to_csv(
    "data/songs/failed_genre_lookups.csv", index=False
)

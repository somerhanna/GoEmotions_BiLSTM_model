import requests, time, json
from functools import lru_cache
import pandas as pd
from unidecode import unidecode
from tqdm import tqdm

FILE_IN = "data/songs/lyrics_fuzzymatched.csv"
FILE_OUT = "data/songs/lyrics_with_tags.csv"
FILE_DEBUG_OUT = "data/songs/failed_tag_lookups.csv"

API_KEY = "82b36dc4aba7706ecde0a72948920b05" # Don't care abt committing this, its free and acct is throwaway
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

RATE_LIMIT_SLEEP = 1.0     # how long to wait after 429
REQUEST_SLEEP = 0.25       # normal pacing delay
MAX_RETRIES = 5            # network/timeout retry attempts

ARTIST_COL = "corrected_artist"
SONG_COL   = "corrected_track"
ALBUM_COL = "Album"

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
                tqdm.write(f"Error code: {error_code}, {data.get('message')}")
                return None, f"api_error_{error_code}"

            tags = data.get("toptags", {}).get("tag")

            # No tags at all
            if not tags:
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

@lru_cache(maxsize=100000)
def get_album_tag(artist, album):
    """Return (tag, status) for album.getTopTags"""

    if not album:
        return None, "no_album"

    params = {
        "method": "album.gettoptags",
        "artist": artist,
        "album": album,
        "api_key": API_KEY,
        "format": "json",
        "autocorrect": 1
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE_URL, params=params)

            if r.status_code == 429:
                rate_limit_backoff()
                continue

            if r.status_code >= 500:
                time.sleep(1)
                continue

            data = r.json()

            # Last.fm error case (album unknown)
            if "error" in data:
                return None, f"api_error_album_{data['error']}"

            tags = data.get("toptags", {}).get("tag")
            if not tags:
                return None, "album_no_tags"

            if isinstance(tags, dict):
                return tags.get("name"), "album_ok"

            if isinstance(tags, list) and len(tags) > 0:
                return tags[0].get("name"), "album_ok"

            return None, "album_no_tags"

        except:
            time.sleep(1)

    return None, "album_network_error"

def get_top_tag_with_fallback(artist, track, album):
    """
    Try track tag first, then album fallback.
    Returns (tag, status)
    """

    # --- Try TRACK tags ---
    tag, status = get_top_tag(artist, track)

    if status == "ok":
        return tag, "track_ok"

    # If track-level tags missing, try ALBUM tags
    if album:
        album_tag, album_status = get_album_tag(artist, album)
        if album_status == "album_ok":
            return album_tag, "album_ok"

    # No tag found at track or album level
    return None, "no_tags"

# Load Data
df = pd.read_csv(FILE_IN)

# Prepopulate needed columns
if "tags" not in df.columns:
    df["tags"] = None

failed_artist_song_pairs = []

# MAIN LOOP
pbar = tqdm(range(len(df)))
for i in pbar:
    artist = str(df.iloc[i][ARTIST_COL])
    album = str(df.iloc[i][ALBUM_COL])
    track  = str(df.iloc[i][SONG_COL])

    pbar.set_description(
        f"{artist[:20]} - {track[:20]} | fails={len(failed_artist_song_pairs)}"
    )

    if not artist or not track:
        df.iloc[i, df.columns.get_loc("tags")] = None
        continue

    tag, status = get_top_tag_with_fallback(artist, track, album)

    if status in ("track_ok", "album_ok"):
        df.iloc[i, df.columns.get_loc("tags")] = tag
    else:
        failed_artist_song_pairs.append((artist, track, status))

    time.sleep(REQUEST_SLEEP)


# SAVE RAW
df.to_csv(FILE_OUT, index=False)

# SAVE FAILURES FOR LATER ANALYSIS
pd.DataFrame(failed_artist_song_pairs, columns=["artist", "track", "reason"]).to_csv(
    FILE_DEBUG_OUT, index=False
)

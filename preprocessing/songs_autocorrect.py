import requests
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from functools import lru_cache
import time

FILE_IN = "data/songs/merged_lyrics.csv"
FILE_OUT = "data/songs/lyrics_fuzzymatched.csv"

API_KEY = "82b36dc4aba7706ecde0a72948920b05" # Don't care abt committing this, its free and acct is throwaway
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

ARTIST_COL = "Artist"
TRACK_COL = "Title"

REQUEST_SLEEP = 0.20

def normalize(s):
    """Lowercase + remove accents + trim whitespace."""
    if not isinstance(s, str):
        return ""
    return unidecode(s.lower().strip())


@lru_cache(maxsize=200000)
def fuzzy_search_lastfm(artist, track):
    """
    Performs fuzzy search on Last.fm using track.search.
    Returns (corrected_artist, corrected_track, status)
    status ∈ {"ok", "no_match", "api_error", "network_error"}
    """

    params = {
        "method": "track.search",
        "track": track,
        "artist": artist,
        "api_key": API_KEY,
        "format": "json",
        "limit": 1
    }

    try:
        r = requests.get(BASE_URL, params=params, timeout=5)

        # Last.fm rate limit
        if r.status_code == 429:
            time.sleep(1)
            return fuzzy_search_lastfm(artist, track)

        if r.status_code != 200:
            return None, None, "api_error"

        data = r.json()

    except:
        return None, None, "network_error"

    matches = (
        data.get("results", {})
            .get("trackmatches", {})
            .get("track", [])
    )

    if not matches:
        return None, None, "no_match"

    # If a single dict
    if isinstance(matches, dict):
        best = matches
    else:
        best = matches[0]

    corrected_artist = best.get("artist")
    corrected_track = best.get("name")

    if(not corrected_artist and not corrected_track):
        return None, None, "no_match"

    exact_match = corrected_artist == artist and corrected_track == track
    status = "exact_match" if exact_match else "ok"

    return corrected_artist, corrected_track, status

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

df = pd.read_csv(FILE_IN)

# Columns to hold fuzzy-corrected names
df["corrected_artist"] = None
df["corrected_track"] = None
df["match_status"] = None

# -------------------------------------------------------
# PROCESS DATA
# -------------------------------------------------------

pbar = tqdm(range(len(df)), desc="Fuzzy matching...")

for i in pbar:

    artist = df.iloc[i][ARTIST_COL]
    track  = df.iloc[i][TRACK_COL]

    if not isinstance(artist, str) or not isinstance(track, str):
        df.iloc[i, df.columns.get_loc("match_status")] = "invalid_input"
        continue

    # Normalized forms for searching
    norm_artist = normalize(artist)
    norm_track  = normalize(track)

    corrected_artist, corrected_track, status = fuzzy_search_lastfm(norm_artist, norm_track)

    df.iloc[i, df.columns.get_loc("corrected_artist")] = corrected_artist
    df.iloc[i, df.columns.get_loc("corrected_track")]  = corrected_track
    df.iloc[i, df.columns.get_loc("match_status")]     = status

    pbar.set_description(f"Status: {status}")

    time.sleep(REQUEST_SLEEP)


# -------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------

df.to_csv(FILE_OUT, index=False)
print(f"\nFuzzy matching complete! Saved to: {FILE_OUT}")

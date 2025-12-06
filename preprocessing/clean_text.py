# preprocessing/clean_text.py
import re

def clean_lyrics(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Remove weird punctuation / symbols
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

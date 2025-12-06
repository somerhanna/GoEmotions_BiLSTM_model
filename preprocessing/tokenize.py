# preprocessing/tokenize.py
from utils.vocab import Vocab

def build_vocab(texts, min_freq=2):
    """
    Build a vocabulary from a list of cleaned texts.
    Only include tokens that appear at least `min_freq` times.
    """
    vocab = Vocab()
    for text in texts:
        tokens = text.split()
        for tok in tokens:
            vocab.add_token(tok)
    vocab.build(min_freq=min_freq)
    return vocab

def encode_text(text, vocab, max_len=300):
    """
    Convert a string into a fixed-length list of token IDs using `vocab`.
    Pads with <pad> or truncates to exactly `max_len` tokens.
    """
    tokens = text.split()
    ids = [vocab.stoi.get(tok, vocab.stoi["<unk>"]) for tok in tokens]

    if len(ids) < max_len:
        ids = ids + [vocab.stoi["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids

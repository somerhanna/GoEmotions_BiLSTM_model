from collections import Counter

class Vocab:
    def __init__(self):
        self.counter = Counter()
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def add_token(self, token):
        self.counter[token] += 1

    def build(self, min_freq):
        for token, count in self.counter.items():
            if count >= min_freq:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self):
        return len(self.itos)

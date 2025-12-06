import torch
from torch.utils.data import Dataset

class LyricsDataset(Dataset):
    def __init__(self, encoded_texts, labels):
        self.x = encoded_texts
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

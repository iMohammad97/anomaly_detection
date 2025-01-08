import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UCR(Dataset):
    def __init__(self, path: str, window_size: int, step_size: int = 1, train: bool = True):
        self.path = path
        self.window_size, self.step_size = window_size, step_size
        if train:
            self.data = self.create_windows('train')
            self.labels = torch.zeros_like(self.data)
        else:
            self.data = self.create_windows('test')
            self.labels = self.create_windows('labels')

    def create_windows(self, tag: str):
        files = [f for f in os.listdir(self.path) if f.endswith(f'{tag}.npy')]
        windows = []
        for file_path in files:
            array = np.load(os.path.join(self.path, file_path))
            for i in range(0, len(array) - self.window_size + 1, self.step_size):
                windows.append(array[i:i + self.window_size])
        windows = np.array(windows) # because it's faster
        return torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UCR(Dataset):
    def __init__(self, path: str, window_size: int, step_size: int = 1, train: bool = True, data_id: int = 0):
        self.path = path
        self.window_size, self.step_size = window_size, step_size
        if train:
            self.data = self.create_windows('train', data_id)
            self.labels = torch.zeros_like(self.data)
        else:
            self.data = self.create_windows('test', data_id)
            self.labels = self.create_windows('labels', data_id)

    def create_windows(self, tag: str, data_id: int):
        files = [f for f in os.listdir(self.path) if f.endswith(f'{tag}.npy')]
        windows = []
        for file_path in files:
            if data_id != 0 and not file_path.startswith(f'{str(data_id)}_'):
                continue
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

def get_dataloaders(path: str, window_size: int, batch_size: int, step_size: int = 1, data_id: int = 0):
    # Create datasets
    train_dataset = UCR(path, window_size, step_size=step_size, train=True, data_id=data_id)
    test_dataset = UCR(path, window_size, step_size=1, train=False, data_id=data_id) # test step size should always be 1
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Yahoo(Dataset):
    def __init__(self, path: str, window_size: int, train: bool, step_size: int = 1, train_split: float = 0.5):
        self.path, self.train, self.split = path, train, train_split
        self.window_size, self.step_size = window_size, step_size
        self.data = self.create_windows('test')
        self.labels = self.create_windows('labels')
        self.normalize = None

    def create_windows(self, tag: str):
        windows = []
        try:
            array = np.load(f'{self.path}_{tag}.npy')
        except:
            raise 'Invalid data_id! This time series does not exist!'
        window_count = len(array) - self.window_size + 1
        if self.train: # Choose the first split
            start_at = 0
            ent_at = int(self.split * window_count)
        else: # Choose the second split
            start_at = int(self.split * window_count) + 1
            ent_at = window_count
        for i in range(start_at, ent_at, self.step_size):
            windows.append(array[i:i + self.window_size])
        windows = np.array(windows) # because it's faster
        return torch.tensor(windows, dtype=torch.float32)

    def set_normalize(self, minimum, maximum):
        self.normalize = lambda x : (x - minimum) / (maximum - minimum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.normalize is not None:
            return self.normalize(self.data[idx]), self.labels[idx]
        return self.data[idx], self.labels[idx]

def get_dataloaders(path: str, benchmark: int, data_id: int, window_size: int, batch_size: int, train_split: float = 0.5, normalize: bool = True, step_size: int = 1, shuffle: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    tags = ['real_', 'synthetic_', 'A3Benchmark-TS', 'A4Benchmark-TS']
    if benchmark in (1, 2, 3, 4):
        path = f'{path}/A{benchmark}Benchmark/{tags[benchmark-1]}{data_id}'
    else:
        raise f'Benchmark {benchmark} does not exist!'
    # Create datasets
    train_dataset = Yahoo(path, window_size, train=True, step_size=step_size, train_split=train_split)
    test_dataset = Yahoo(path, window_size, train=False, step_size=1, train_split=train_split) # test step size should always be 1
    # Normalize
    if normalize:
        minimum, maximum = torch.min(train_dataset.data), torch.max(train_dataset.data)
        train_dataset.set_normalize(minimum=minimum, maximum=maximum)
        test_dataset.set_normalize(minimum=minimum, maximum=maximum)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

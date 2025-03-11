import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utilities.preprocess import preprocess_SMD


class NAB(Dataset):
    def __init__(self, path: str, window_size: int, train: bool, step_size: int = 1, train_split: float = 0.5):
        self.path, self.train, self.split = path, train, train_split
        self.window_size, self.step_size = window_size, step_size
        self.data = self.create_windows('test')
        self.labels = self.create_windows('label')

    def create_windows(self, tag: str):
        windows = []
        try:
            array = np.load(f'{self.path}_{tag}.npy')
        except:
            raise 'This time series does not exist!'
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

def get_dataloaders(path: str, time_series: str, window_size: int, batch_size: int, train_split: float = 0.5, step_size: int = 1, shuffle: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    possible_series = ['artificialWithAnomaly_art_daily_flatmiddle',
                       'artificialWithAnomaly_art_daily_jumpsdown',
                       'artificialWithAnomaly_art_daily_jumpsup',
                       'artificialWithAnomaly_art_increase_spike_density',
                       'realAWSCloudwatch_ec2_cpu_utilization_24ae8d',
                       'realKnownCause_ambient_temperature_system_failure',
                       'realTraffic_occupancy_6005']
    if time_series in possible_series:
        path = f'{path}/{time_series}'
    else:
        print('\tPossible options include:')
        for ts in possible_series:
            print(ts)
        raise f'Time series {time_series} does not exist!'
    # Create datasets
    train_dataset = NAB(path, window_size, train=True, step_size=step_size, train_split=train_split)
    test_dataset = NAB(path, window_size, train=False, step_size=1, train_split=train_split) # test step size should always be 1
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

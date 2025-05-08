import requests
import zipfile
import io
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TSB(Dataset):
    def __init__(self, data_id: int, window_size: int, train: bool, first_start: int, first_end: int, step_size: int = 1, normalization=None):
        self.path = 'data/TSB-AD-U'
        self.normalize, self.normalization = None, normalization

        # Download the dataset if necessary
        if not os.path.exists(self.path):
            self.download()

        self.window_size, self.step_size = window_size, step_size
        self.data, self.labels = self.get_windows(data_id, train, first_start, first_end)

    @staticmethod
    def download():
        url = 'https://www.thedatum.org/datasets/TSB-AD-U.zip'
        response = requests.get(url)
        response.raise_for_status() # Ensure we notice bad responses
        # Open the ZIP file from the downloaded content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('data')  # Extract to the specified folder
        print(f'Downloaded the dataset from {url}')


    def get_windows(self, data_id: int, train: bool, first_start: int, first_end: int):
        files = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        for file_path in files:
            if file_path.startswith(f'{str(data_id).zfill(3)}_'):
                df = pd.read_csv(os.path.join(self.path, file_path))
                array, label = np.array(df['Data'])[first_start:first_end], np.array(df['Label'])[first_start:first_end]
                # Split the data
                length = len(array)
                train_array, test_array = array[:length//2], array[length//2:]
                train_label, test_label = label[:length//2], label[length//2:]
                # Set normalization
                if self.normalization == 'MinMax':
                    self.set_minmax_normalize(np.min(train_array), np.max(train_array))
                elif self.normalization == 'Z':
                    self.set_z_normalization(np.mean(train_array), np.std(train_array))
                if train:
                    data = self.create_windows(train_array)
                    label = self.create_windows(train_label)
                else:
                    data = self.create_windows(test_array)
                    label = self.create_windows(test_label)
                return data, label
        raise f'Time Series {data_id} does not exist!'

    def create_windows(self, array):
        windows = []
        window_count = len(array) - self.window_size + 1
        for i in range(0, window_count, self.step_size):
            windows.append(array[i:i + self.window_size])
        windows = np.array(windows)  # because it's faster
        return torch.tensor(windows, dtype=torch.float32).unsqueeze(2)

    def set_minmax_normalize(self, min_val, max_val):
        self.normalize = lambda x: (x - min_val) / (max_val - min_val)

    def set_z_normalization(self, mu, sigma):
        self.normalize = lambda x: (x - mu) / sigma

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.normalize is not None:
            return self.normalize(self.data[idx]), self.labels[idx]
        return self.data[idx], self.labels[idx]



def get_dataloaders(data_id: int, window_size: int, first_start: int, first_end: int, batch_size: int = 256, normalization: str ='Z', step_size: int = 1, shuffle: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    # Create datasets
    train_dataset = TSB(data_id, window_size, first_start=first_start, first_end=first_end, step_size=step_size, train=True, normalization=normalization)
    test_dataset = TSB(data_id, window_size, first_start=first_start, first_end=first_end, train=False, normalization=normalization) # test step size should always be 1
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

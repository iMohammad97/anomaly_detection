import requests
import zipfile
import io
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TSB_UAD(Dataset):
    def __init__(self, window_size: int, data_id: int, train: bool, step_size: int = 1, train_split: float = 0.5):
        self.path = 'data/TSB-AD-U'
        self.normalize = None

        # Download the dataset if necessary
        if not os.path.exists(self.path):
            self.download()

        self.window_size, self.step_size = window_size, step_size
        self.train, self.split = train, train_split

        self.data, self.labels = self.get_windows(data_id)
            

    def download(self):
        url = 'https://www.thedatum.org/datasets/TSB-AD-U.zip'
        response = requests.get(url)
        response.raise_for_status() # Ensure we notice bad responses

        # Open the ZIP file from the downloaded content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('data')  # Extract to the specified folder
        print(f'Donwloaded the dataset from {url}')


    def get_windows(self, data_id):
        files = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        for file_path in files:
            if file_path.startswith(f'{str(data_id).zfill(3)}_'):
                df = pd.read_csv(os.path.join(self.path, file_path))
                data = self.create_windows(np.array(df['Data']))
                label = self.create_windows(np.array(df['Label']))
                return data, label
        raise 'Time Series doesn\'t exist!'


    def create_windows(self, array):
        windows = []
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
        return torch.tensor(windows, dtype=torch.float32).unsqueeze(2)

    def set_normalize(self, minimum, maximum):
        self.normalize = lambda x : (x - minimum) / (maximum - minimum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.normalize is not None:
            return self.normalize(self.data[idx]), self.labels[idx]
        return self.data[idx], self.labels[idx]

def get_dataloaders(window_size: int, data_id: int, batch_size: int = 256, train_split: float = 0.5, step_size: int = 1, shuffle: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    # Create datasets
    train_dataset = TSB_UAD(window_size, data_id, step_size=step_size, train=True, train_split=train_split)
    test_dataset = TSB_UAD(window_size, data_id, step_size=1, train=False, train_split=train_split) # test step size should always be 1
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    

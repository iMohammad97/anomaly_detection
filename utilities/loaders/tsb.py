import requests
import zipfile
import io
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TSB(Dataset):
    def __init__(self, data_id: int, window_size: int, train: bool, step_size: int = 1, normalization=None):
        self.path = 'data/TSB-AD-U'
        self.normalize, self.normalization = None, normalization

        # Download the dataset if necessary
        if not os.path.exists(self.path):
            self.download()

        self.window_size, self.step_size = window_size, step_size
        self.data, self.labels = self.get_windows(data_id, train)

    @staticmethod
    def download():
        url = 'https://www.thedatum.org/datasets/TSB-AD-U.zip'
        response = requests.get(url)
        response.raise_for_status() # Ensure we notice bad responses
        # Open the ZIP file from the downloaded content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('data')  # Extract to the specified folder
        print(f'Downloaded the dataset from {url}')


    def get_windows(self, data_id: int, train: bool):
        files = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        for file_path in files:
            if file_path.startswith(f'{str(data_id).zfill(3)}_'):
                split = int(file_path.split('.')[0].split('_')[-3])
                df = pd.read_csv(os.path.join(self.path, file_path))
                array, label = np.array(df['Data']), np.array(df['Label'])
                # Split the data
                train_array, test_array = array[:split], array[split:]
                train_label, test_label = label[:split], label[split:]
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



def get_dataloaders(data_id: int, window_size: int, batch_size: int = 256, normalization: str ='Z', step_size: int = 1, shuffle: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    # Create datasets
    train_dataset = TSB(data_id, window_size, step_size=step_size, train=True, normalization=normalization)
    test_dataset = TSB(data_id, window_size, train=False, normalization=normalization) # test step size should always be 1
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Exclude the following time series for unsupervised learning
skips = [9, 10, 91, 92, 110, 113, 114, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
         165, 166, 167, 168, 188, 216, 217, 218, 220, 221, 257, 258, 259, 271, 277, 278, 280, 282, 287, 288, 289, 290,
         291, 292, 293, 294, 296, 297, 298, 299, 301, 553, 556, 561, 565, 568, 569, 572, 574, 577, 580, 586, 589, 591,
         592, 593, 595, 604, 607, 608, 609, 612, 615, 619, 622, 623, 628, 629, 631, 633, 636, 637, 640, 641, 642, 646,
         651, 652, 653, 654, 655, 660, 661, 662, 665, 667, 668, 669, 670, 677, 679, 682, 683, 685, 687, 688, 691, 692,
         697, 698, 699, 700, 702, 706, 707, 708, 709, 710, 711, 712, 713, 714, 717, 718, 720, 722, 726, 727, 728, 733,
         735, 737, 738, 740, 743, 745, 746, 748, 749, 751, 752, 753, 754, 756, 757, 758, 759, 760, 761, 763, 765, 766,
         768, 769, 770, 771, 772, 773, 779, 783, 789, 790, 793, 795, 796, 798, 801, 802, 803, 804, 805, 808, 809]

def find_supervised_time_series(dataset_path: str):
    files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    supervised_ts = []
    for file_path in files:
        split = int(file_path.split('.')[0].split('_')[-3])
        first = int(file_path.split('.')[0].split('_')[-1])
        if first < split:
            print(file_path)
            supervised_ts.append(file_path.split('.')[0].split('_')[0])
    len(supervised_ts)
    return supervised_ts


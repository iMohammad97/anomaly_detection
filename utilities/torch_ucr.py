import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UCR(Dataset):
    """
    Loads the train/test/labels .npy files from a given path,
    splits them into windows, and returns them in a dataset-friendly format.
    If your dataset is huge, you might consider a streaming approach.
    """
    def __init__(self, path: str, window_size: int, step_size: int = 1, train: bool = True, data_id: int = 0):
        self.path = path
        self.window_size = window_size
        self.step_size = step_size
        self.train = train
        self.data_id = data_id

        # We pre-load the entire windows into self.data (and labels if test).
        # For extremely large data, consider writing a streaming generator.
        if train:
            self.data, self.labels = self.create_windows(tag='train')
        else:
            self.data, self.labels = self.create_windows(tag='test')
            # For anomalies, we read "labels" separately
            label_data, _ = self.create_windows(tag='labels')
            self.labels = label_data

    def create_windows(self, tag: str):
        """
        Reads all *.npy files in self.path that end with `tag + ".npy"`.
        If data_id != 0, only load files that start with data_id_.
        Then slide a window of length self.window_size with step=self.step_size.
        Returns: (windows_tensor, dummy_label_tensor)  # label is zero if tag != 'labels'
        """
        files = [
            f for f in os.listdir(self.path)
            if f.endswith(f"{tag}.npy")
        ]
        all_windows = []
        for file_path in files:
            # If data_id != 0, skip files that don't start with f"{data_id}_"
            if self.data_id != 0 and not file_path.startswith(f"{str(self.data_id)}_"):
                continue

            array = np.load(os.path.join(self.path, file_path))

            # Slide over the entire array
            for i in range(0, len(array) - self.window_size + 1, self.step_size):
                window = array[i:i + self.window_size]
                all_windows.append(window)

        if len(all_windows) == 0:
            # If no files or no windows found, create dummy array
            all_windows = np.zeros((0, self.window_size))

        all_windows = np.array(all_windows, dtype=np.float32)
        # Expand last dimension if your data is 1D: shape => (N, window_size, 1)
        if len(all_windows.shape) == 2:
            all_windows = all_windows[..., None]  # (N, window_size, 1)

        # Return data + a zero label array if tag != 'labels'
        if tag == 'labels':
            return torch.tensor(all_windows), None
        else:
            dummy_labels = torch.zeros_like(torch.tensor(all_windows))
            return torch.tensor(all_windows), dummy_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns tuple (window, label).
        - window shape: (window_size, n_features)
        - label shape:  (window_size, n_features) if "test" and we loaded anomaly labels,
                        or zeros otherwise
        """
        return self.data[idx], self.labels[idx]



def get_dataloaders(path: str, window_size: int, batch_size: int, step_size: int = 1, data_id: int = 0, shuffle: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    # Create datasets
    train_dataset = UCR(path, window_size, step_size=step_size, train=True, data_id=data_id)
    test_dataset = UCR(path, window_size, step_size=1, train=False, data_id=data_id) # test step size should always be 1
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

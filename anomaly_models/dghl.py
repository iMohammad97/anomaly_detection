import torch
import random
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import  DataLoader, TensorDataset
from time import time
from anomaly_models.dghlfolder import model
from anomaly_models.dghlfolder.experiment import train_DGHL
from anomaly_models.dghlfolder.utils import basic_mc



def ready_data(train_data, test_data):
    train_data = train_data[:, None, :]
    test_data = test_data[:, None, :]

    train_max = train_data.max(axis=0, keepdims=True)

    train_data = train_data / train_max
    test_data = test_data / train_max

    train_mask = np.ones(train_data.shape)
    test_mask = np.ones(test_data.shape)

    train_data = [train_data]
    train_mask = [train_mask]
    test_data = [test_data]
    test_mask = [test_mask]

    # Using zeros as placeholder labels
    test_labels = np.zeros((test_data[0].shape[0], 1))
    test_labels = [test_labels]

    return train_data, train_mask, test_data, test_mask, test_labels

def run_dghl_model(data, train_size=1000, predict_size=300):
    total_size = len(data)
    predictions = []

    index=data.index
    data=data.to_numpy().reshape(-1, 1).astype(np.float32)
    
    for start_idx in range(0, total_size - train_size, predict_size):
        end_idx = start_idx + train_size
        predict_idx = end_idx + predict_size

        if end_idx > total_size:
            break

        train_data = data[0:end_idx]
        if predict_idx > total_size:
            test_data = data[end_idx:end_idx]
        else:
            test_data = data[end_idx:predict_idx]

        train_data, train_mask, test_data, test_mask, test_labels = ready_data(train_data, test_data)

        torch.manual_seed(40)
        random.seed(40)
        np.random.seed(40)
        mc = basic_mc(train_data[0].shape[2], random_seed=1)
        root_dir = f'./dghlresults/DGHL-nomask-smdlabels/dghl'

        scores = train_DGHL(mc=mc, train_data=train_data, test_data=test_data,
                            train_mask=train_mask, test_labels=test_labels, test_mask=test_mask,
                            entities=['entity'], make_plots=False, root_dir=root_dir)
        print(scores)
        predictions.extend(scores)
        print(predictions)
    import pandas as pd
    df = pd.DataFrame(predictions, columns=['predictions'])
    df.index = index
    return predictions

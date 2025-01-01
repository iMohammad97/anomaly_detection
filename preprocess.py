from sklearn.preprocessing import MinMaxScaler
import ast
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from shutil import copyfile
import matplotlib.pyplot as plt


def preprocess_UCR(dataset_raw_dir,dataset_processed_dir):
    file_list = os.listdir(dataset_raw_dir)
    dataset_categories = []
    base_directory = dataset_processed_dir
    for filename in file_list:
        if not filename.endswith('.txt'):
            continue

        filename_parts = filename.split('.')[0].split('_')
        try:
            dataset_number = int(filename_parts[0])
            dataset_category = filename_parts[3]
            dataset_categories.append(dataset_category)
            new_directory_path = os.path.join(base_directory, dataset_category)
            os.makedirs(new_directory_path, exist_ok=True)
            train_range, start_anomaly, end_anomaly = map(int, filename_parts[-3:])
        except ValueError:
            print(f"Skipping file {filename}: Invalid filename format.")
            continue

        file_path = os.path.join(dataset_raw_dir, filename)
        try:
            raw_data = np.genfromtxt(file_path, dtype=np.float64, delimiter=',')
        except IOError:
            print(f"Error reading file {file_path}.")
            continue

        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(raw_data.reshape(-1, 1))

        training_data = normalized_data[:train_range]
        testing_data = normalized_data[train_range:]
        anomaly_labels = np.zeros_like(testing_data)
        anomaly_labels[start_anomaly - train_range:end_anomaly - train_range] = 1

        data_splits = {'train': training_data, 'test': testing_data, 'labels': anomaly_labels}
        for split_name in data_splits:
            np.save(os.path.join(new_directory_path, f'{dataset_number}_{split_name}.npy'), data_splits[split_name])

def preprocess_SMD(dataset_raw_dir, dataset_processed_dir):
    if not os.path.exists(dataset_processed_dir):
        os.makedirs(dataset_processed_dir)

    train_files = os.listdir(os.path.join(dataset_raw_dir, "train"))

    for filename in train_files:
        if filename.endswith('.txt'):
            dataset = filename.strip('.txt')

            # Process training data
            train_data = np.genfromtxt(os.path.join(dataset_raw_dir, 'train', filename), dtype=np.float64, delimiter=',')
            np.save(os.path.join(dataset_processed_dir, f"{dataset}_train.npy"), train_data)

            # Process test data
            test_data = np.genfromtxt(os.path.join(dataset_raw_dir, 'test', filename), dtype=np.float64, delimiter=',')
            np.save(os.path.join(dataset_processed_dir, f"{dataset}_test.npy"), test_data)

            # Process labels
            labels_data = np.zeros(test_data.shape)
            interpretation_label_folder = os.path.join(dataset_raw_dir, 'interpretation_label')
            with open(os.path.join(interpretation_label_folder, filename), "r") as f:
                lines = f.readlines()
            for line in lines:
                pos, values = line.split(':')[0], line.split(':')[1].split(',')
                start, end, indices = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i) - 1 for i in values]
                labels_data[start - 1:end-1, indices] = 1
            np.save(os.path.join(dataset_processed_dir, f"{dataset}_labels.npy"), labels_data)


def preprocess_WADI(dataset_raw_dir, dataset_processed_dir):
    if not os.path.exists(dataset_processed_dir):
        os.makedirs(dataset_processed_dir)
    test = pd.read_csv(os.path.join(dataset_raw_dir, 'WADI_attackdataLABLE.csv'),header=1)
    train = pd.read_csv(os.path.join(dataset_raw_dir, 'WADI_14days_new.csv'))
    train.dropna(how='all', inplace=True);train.fillna(0, inplace=True)
    test.dropna(how='all', inplace=True);test.fillna(0, inplace=True)
    for date in train['Date'].unique():
      df_filtered = train[train['Date'] == date]
      df_filtered = df_filtered.drop(['Date', 'Time'], axis=1)
      df_filtered.set_index('Row', inplace=True)
      array_data = df_filtered.to_numpy()
      np.save(os.path.join(dataset_processed_dir, f"{date.replace('/','-')}_train.npy"), array_data)
    for date in test['Date '].unique():
      if '/' not in str(date):
        continue
      df_filtered = test[test['Date '] == date]
      labels = test['Attack LABLE (1:No Attack, -1:Attack)'].to_numpy()
      df_filtered = df_filtered.drop(['Date ', 'Time','Attack LABLE (1:No Attack, -1:Attack)'], axis=1)
      print(date)
      df_filtered.set_index('Row ', inplace=True)
      array_data = df_filtered.to_numpy()
      np.save(os.path.join(dataset_processed_dir, f"{date.replace('/','-')}_test.npy"), array_data)
      np.save(os.path.join(dataset_processed_dir, f"{date.replace('/','-')}_labels.npy"), labels)

def preprocess_SMAPMSL(dataset_raw_dir, dataset_processed_dir):
    scaler = MinMaxScaler()
    if not os.path.exists(dataset_processed_dir):
        os.makedirs(dataset_processed_dir)
    labels_df = pd.read_csv(os.path.join(dataset_raw_dir, 'labeled_anomalies.csv'))
    filenames = labels_df['chan_id'].values.tolist()

    for chan in filenames:
        # Load and scale training data
        train = np.load(f'{dataset_raw_dir}/train/{chan}.npy')
        train = scaler.fit_transform(train)

        # Load and scale test data
        test = np.load(f'{dataset_raw_dir}/test/{chan}.npy')
        test = scaler.fit_transform(test)

        # Save scaled data
        np.save(f'{dataset_processed_dir}/{chan}_train.npy', train)
        np.save(f'{dataset_processed_dir}/{chan}_test.npy', test)

        # Initialize labels array
        labels = np.zeros(test.shape[0])

        # Filter rows for the current chan and extract anomaly sequences
        anomaly_sequences_strings = labels_df[labels_df['chan_id'] == chan]['anomaly_sequences'].tolist()

        for seq_str in anomaly_sequences_strings:
            # Convert the string to a list of tuples
            seq = ast.literal_eval(seq_str)
            # Process each tuple in the list
            for start, end in seq:
                labels[start:end+1] = 1  # Marking the anomalies
        # Save labels
        np.save(f'{dataset_processed_dir}/{chan}_labels.npy', labels)

def preprocess_yahoo(input_dir, output_dir):
    # List all entries in the given raw dataset directory
    dir_list = os.listdir(input_dir)

    # Iterate over each entry in the directory list
    for dir_name in dir_list:
        # Construct the full path to the current directory
        full_dir_path = os.path.join(input_dir, dir_name)

        # Check if the current entry is a directory and contains 'benchmark' in its name
        if os.path.isdir(full_dir_path) and 'Benchmark' in dir_name:
            print(f"Processing directory: {dir_name}")

            # Create a directory for each dir_name within the processed directory
            dir_output_path = os.path.join(output_dir, dir_name)
            os.makedirs(dir_output_path, exist_ok=True)

            # List all files in the benchmark directory
            for file_name in os.listdir(full_dir_path):
                if file_name in ["A1Benchmark_all.csv", "A2Benchmark_all.csv", "A3Benchmark_all.csv", "A4Benchmark_all.csv"]:
                    continue

                # Construct the full path to the current file
                full_file_path = os.path.join(full_dir_path, file_name)

                # Check if the current entry is a file
                if os.path.isfile(full_file_path):
                    print(f"Processing file: {file_name}")

                    # Load the CSV file using pandas
                    df = pd.read_csv(full_file_path)

                    # Rename columns if necessary
                    if 'anomaly' in df.columns:
                        df.rename(columns={'anomaly': 'is_anomaly'}, inplace=True)
                    if 'timestamps' in df.columns:
                        df.rename(columns={'timestamps': 'timestamp'}, inplace=True)

                    # Save relevant columns to a numpy array (.npy) within the dir_name directory
                    npy_filename_values = os.path.join(dir_output_path, f"{file_name.split('.')[0]}_test.npy")
                    np.save(npy_filename_values, df[[ 'value']].to_numpy())

                    npy_filename_labels = os.path.join(dir_output_path, f"{file_name.split('.')[0]}_labels.npy")
                    np.save(npy_filename_labels, df['is_anomaly'].to_numpy())


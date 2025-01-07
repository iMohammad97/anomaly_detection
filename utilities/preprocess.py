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


def preprocess_UCR(
        dataset_raw_dir,
        dataset_processed_dir,
        split_by_source=False,
        split_by_anomaly_type=False,
        anomalies_type_csv=None
):
    """
    Preprocess .txt files into .npy (train/test/labels).
    Can split by 'source' (extracted from filename) or by 'anomaly_type' (from CSV), or both.

    :param dataset_raw_dir: Directory containing .txt files.
    :param dataset_processed_dir: Directory to store output .npy files.
    :param split_by_source: If True, separate subfolders by the 'source' in filename_parts[3].
    :param split_by_anomaly_type: If True, separate subfolders by the anomaly type from CSV (column 'anomaly_type_2').
    :param anomalies_type_csv: Path to CSV file with columns 'name' and 'anomaly_type_2'.
                               'name' should match the numeric ID in the .txt filename (e.g. '100' for '100_whatever.txt').
    """

    # -- 1. If split_by_anomaly_type is True, load the CSV into a dictionary for quick look-up --
    anomaly_dict = {}
    if split_by_anomaly_type:
        if not anomalies_type_csv:
            raise ValueError("If split_by_anomaly_type=True, you must provide anomalies_type_csv.")
        df = pd.read_csv(anomalies_type_csv, sep=';', dtype=str)
        # Ensure it has the required columns
        if not {'name', 'anomaly_type_2'}.issubset(df.columns):
            raise ValueError("CSV must contain columns 'name' and 'anomaly_type_2'.")
        # Build a dict: {"100": "spike", "101": "flatline", ...}
        anomaly_dict = dict(zip(df['name'], df['anomaly_type_2']))

    file_list = os.listdir(dataset_raw_dir)
    base_directory = dataset_processed_dir

    # Keep track of categories if needed (optional)
    dataset_categories = []

    for filename in file_list:
        if not filename.endswith('.txt'):
            continue

        filename_parts = filename.split('.')[0].split('_')
        # Example filename: "100_something_something_source_150_160_170.txt"
        # filename_parts = ["100", "something", "something", "source", "150", "160", "170"]

        try:
            # The first element is the numeric dataset ID
            dataset_number = int(filename_parts[0])  # e.g. 100

            # Example: train_range, start_anomaly, end_anomaly in the *last three* parts
            train_range, start_anomaly, end_anomaly = map(int, filename_parts[-3:])

            # If splitting by 'source', we assume it's at filename_parts[3]
            dataset_source = None
            if split_by_source:
                dataset_source = filename_parts[3]
                dataset_categories.append(dataset_source)

        except ValueError:
            print(f"Skipping file {filename}: Invalid filename format.")
            continue

        # Build full path to the raw .txt
        file_path = os.path.join(dataset_raw_dir, filename)

        # Read data
        try:
            raw_data = np.genfromtxt(file_path, dtype=np.float64, delimiter=',')
        except IOError:
            print(f"Error reading file {file_path}.")
            continue

        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(raw_data.reshape(-1, 1))

        # Split into train/test
        training_data = normalized_data[:train_range]
        testing_data = normalized_data[train_range:]

        # Build labels array
        anomaly_labels = np.zeros_like(testing_data)
        anomaly_labels[start_anomaly - train_range:end_anomaly - train_range] = 1

        # Prepare dictionary of splits
        data_splits = {
            'train': training_data,
            'test': testing_data,
            'labels': anomaly_labels
        }

        # ----- DETERMINE OUTPUT DIRECTORY based on the chosen modes ------
        #
        #   1) If neither split_by_source nor split_by_anomaly_type => store directly in dataset_processed_dir
        #   2) If split_by_source => create a subdirectory "dataset_source"
        #   3) If split_by_anomaly_type => create a subdirectory named after the anomaly type
        #   4) If BOTH => create nested subdirectories "dataset_source / anomaly_type"
        #
        new_directory_path = base_directory  # default = dataset_processed_dir

        # (A) If split_by_source, create subdir
        if split_by_source and dataset_source is not None:
            new_directory_path = os.path.join(new_directory_path, dataset_source)

        # (B) If split_by_anomaly_type, try to look up anomaly type from anomaly_dict
        if split_by_anomaly_type:
            str_dataset_number = str(dataset_number)  # must match the CSV's 'name' as a string
            if str_dataset_number not in anomaly_dict:
                # If we want to skip these, do so:
                print(f"Skipping file {filename}: ID '{str_dataset_number}' not found in CSV.")
                continue
            anomaly_type_val = anomaly_dict[str_dataset_number]  # e.g. "spike"
            # create subdir from anomaly_type
            new_directory_path = os.path.join(new_directory_path, anomaly_type_val)

        # Make sure the final directory exists
        os.makedirs(new_directory_path, exist_ok=True)

        # ----- SAVE the .npy files ------
        # We'll always name them "<dataset_number>_<split_name>.npy"
        for split_name, split_data in data_splits.items():
            save_path = os.path.join(
                new_directory_path,
                f"{dataset_number}_{split_name}.npy"
            )
            np.save(save_path, split_data)

    # Optional: If you need to know the categories used (when split_by_source),
    # you could return them or do further processing:
    # return dataset_categories


def preprocess_SMD(dataset_raw_dir, dataset_processed_dir):
    if not os.path.exists(dataset_processed_dir):
        os.makedirs(dataset_processed_dir)

    train_files = os.listdir(os.path.join(dataset_raw_dir, "train"))

    for filename in train_files:
        if filename.endswith('.txt'):
            dataset = filename.strip('.txt')

            # Process training data
            train_data = np.genfromtxt(os.path.join(dataset_raw_dir, 'train', filename), dtype=np.float64,
                                       delimiter=',')
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
                labels_data[start - 1:end - 1, indices] = 1
            np.save(os.path.join(dataset_processed_dir, f"{dataset}_labels.npy"), labels_data)


def preprocess_WADI(dataset_raw_dir, dataset_processed_dir):
    if not os.path.exists(dataset_processed_dir):
        os.makedirs(dataset_processed_dir)
    test = pd.read_csv(os.path.join(dataset_raw_dir, 'WADI_attackdataLABLE.csv'), header=1)
    train = pd.read_csv(os.path.join(dataset_raw_dir, 'WADI_14days_new.csv'))
    train.dropna(how='all', inplace=True);
    train.fillna(0, inplace=True)
    test.dropna(how='all', inplace=True);
    test.fillna(0, inplace=True)
    for date in train['Date'].unique():
        df_filtered = train[train['Date'] == date]
        df_filtered = df_filtered.drop(['Date', 'Time'], axis=1)
        df_filtered.set_index('Row', inplace=True)
        array_data = df_filtered.to_numpy()
        np.save(os.path.join(dataset_processed_dir, f"{date.replace('/', '-')}_train.npy"), array_data)
    for date in test['Date '].unique():
        if '/' not in str(date):
            continue
        df_filtered = test[test['Date '] == date]
        labels = test['Attack LABLE (1:No Attack, -1:Attack)'].to_numpy()
        df_filtered = df_filtered.drop(['Date ', 'Time', 'Attack LABLE (1:No Attack, -1:Attack)'], axis=1)
        print(date)
        df_filtered.set_index('Row ', inplace=True)
        array_data = df_filtered.to_numpy()
        np.save(os.path.join(dataset_processed_dir, f"{date.replace('/', '-')}_test.npy"), array_data)
        np.save(os.path.join(dataset_processed_dir, f"{date.replace('/', '-')}_labels.npy"), labels)


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
                labels[start:end + 1] = 1  # Marking the anomalies
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
                if file_name in ["A1Benchmark_all.csv", "A2Benchmark_all.csv", "A3Benchmark_all.csv",
                                 "A4Benchmark_all.csv"]:
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
                    np.save(npy_filename_values, df[['value']].to_numpy())

                    npy_filename_labels = os.path.join(dir_output_path, f"{file_name.split('.')[0]}_labels.npy")
                    np.save(npy_filename_labels, df['is_anomaly'].to_numpy())

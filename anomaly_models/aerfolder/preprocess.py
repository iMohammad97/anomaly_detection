import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = 'UCR/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData'
SAVE_TO = 'UCR/AER'

def build_df(data, start=0):
    index = np.array(range(start, start + len(data)))
    step = 300
    initial_time = 1222819200
    timestamp = index * step + initial_time

    if len(data.shape) > 1 and data.shape[1] > 1:
        print("MULTIVARIATE")
        df = pd.DataFrame(data)
        df['timestamp'] = timestamp
    else:
        df = pd.DataFrame({'timestamp': timestamp, 'value': data.reshape(-1, )})

    df['timestamp'] = df['timestamp'].astype('int64')
    return df

def create_csv_files():
    files = os.listdir(DATA_PATH)
    file_names, train_sizes, intervals = [], [], []

    for file in tqdm(files):
        file_num_str, _, _, file_name, train_size_str, begin_str, end_str = file.split("_")

        train_size, begin_anomaly = int(train_size_str), int(begin_str)
        end_anomaly = int(end_str.split('.')[0])
        file_name = file_num_str + "-" + file_name

        # get timestamp from data
        df = build_df(np.loadtxt(os.path.join(DATA_PATH, file)))
        begin_anomaly = int(df.timestamp.iloc[begin_anomaly])
        end_anomaly = int(df.timestamp.iloc[end_anomaly])

        # train - test split
        train_df = df.iloc[: train_size]
        test_df = df.iloc[train_size:]

        # save file
        train_df.to_csv(SAVE_TO + '/{}-train.csv'.format(file_name), index=False)
        test_df.to_csv(SAVE_TO + '/{}-test.csv'.format(file_name), index=False)
        df.to_csv(SAVE_TO + '/{}.csv'.format(file_name), index=False)

        file_names.append(file_name)
        train_sizes.append(train_size)
        intervals.append([begin_anomaly, end_anomaly])

    return file_names, train_sizes, intervals


def save_anomalies():
    file_names, train_sizes, intervals = create_csv_files()
    rows = []
    for index, file_name in enumerate(file_names):
        row = [file_name, [intervals[index]]]
        rows.append(row)

    # save anomalies

    with open(SAVE_TO + '/anomalies.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == '__main__':
    save_anomalies()

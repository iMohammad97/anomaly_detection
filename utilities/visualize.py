import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import pandas as pd
from datetime import datetime
import seaborn as sns


def visulizeUCR(inputpreprocessdir,outputfile):

    data_list = []

    for file in sorted(os.listdir(inputdir)):
        filepath = os.path.join(inputdir+ '/', file)
        if 'train' in filepath.lower():
            data = np.load(filepath)
            print("Data from file '{}': ".format(filepath))
            data_list.append(data)
            f2 = filepath.replace('train', 'test')
            data_test = np.load(f2)
            print("Data from file '{}': ".format(f2))
            data_list.append(data_test)
            f3 = filepath.replace('train', 'labels')
            data_labels = np.load(f3)
            print("Data from file '{}': ".format(f3))
            data_list.append(data_labels)

    # Create a PDF file with wider pages
    pdf_filename = outputfile
    with PdfPages(pdf_filename, keep_empty=False) as pdf:
        # Plot each data and save to the PDF
        for i in range(0, len(data_list), 3):
            plt.figure(figsize=(100, 5))  # Set figure width to 15 inches
            train_data = data_list[i]
            test_data = data_list[i + 1]
            labels_data = data_list[i + 2]

            # Plotting train data in blue
            plt.plot(train_data, label='Train Data', color='blue')

            # Create a mask based on label values
            mask = labels_data == 1
            plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), test_data, color='green', label='Test Data')
            plt.scatter(np.where(mask)[0] + len(train_data), test_data[mask], color='red', marker='x', label='Test Data (Label=1)')

            plt.xlabel('Data Index')
            plt.ylabel('Data Value')
            plt.title('Train and Test Data')
            plt.legend()
            pdf.savefig()  # Save the current figure into a pdf page
            plt.close()



def visulizeNumenta(inputrawdir,outputfile)

# Set the directory paths
data_directory = inputrawdir + '/data'
labels_file_path = inputrawdir+'/labels/combined_labels.json'
labels_interval_file_path = inputrawdir+'/labels/combined_windows.json'
pdf_filename = outputfile

# Read the JSON file with labels
with open(labels_file_path, 'r') as file:
    labels_data = json.load(file)

# Read the JSON file with interval labels
with open(labels_interval_file_path, 'r') as file:
    labels_interval_data = json.load(file)

os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)

with PdfPages(pdf_filename) as pdf:

  # Iterate through each subdirectory in data_directory
  for subdirectory in os.listdir(data_directory):
      subdirectory_path = os.path.join(data_directory, subdirectory)

      # Check if the path is a directory
      if os.path.isdir(subdirectory_path):
          # Iterate through each file in the subdirectory
          for file_name in os.listdir(subdirectory_path):
              file_path = os.path.join(subdirectory_path, file_name)

              # Check if the file is a CSV file
              if file_name.endswith('.csv'):
                  # Check if the filename is in the labels data
                  df = pd.read_csv(file_path, parse_dates=['timestamp'])
                  print(df)
                  # Plot all values in the DataFrame
                  plt.figure(figsize=(10, 6))
                  plt.plot(df['timestamp'], df['value'], label=f'{subdirectory}/{file_name} - All Values')
                  if f'{subdirectory}/{file_name}' in labels_data:

                      # Get the timestamps from the labels data
                      timestamps = labels_data[f'{subdirectory}/{file_name}']

                      # Add scatter points for each timestamp in the labels
                      for timestamp in timestamps:
                          timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

                          # Find the corresponding value in the DataFrame
                          value = df.loc[df['timestamp'] == timestamp_dt, 'value'].values[0]

                          # Plot the scatter point without label and in red color
                          plt.scatter(timestamp_dt, value, color='red', marker='x')

                      # Get the intervals from the interval labels data
                      intervals = labels_interval_data.get(f'{subdirectory}/{file_name}', [])

                      # Add scatter points for each interval in the interval labels
                      for interval in intervals:
                          start_dt = datetime.strptime(interval[0], '%Y-%m-%d %H:%M:%S.%f')
                          end_dt = datetime.strptime(interval[1], '%Y-%m-%d %H:%M:%S.%f')
                          plt.axvspan(start_dt, end_dt, color='orange', alpha=0.3)

                      # Customize the plot for each file
                  plt.title(f'Plot and Scatter of Labels - {subdirectory}/{file_name}')
                  plt.xlabel('Timestamp')
                  plt.ylabel('Value')
                  plt.xticks(rotation=45)
                  plt.legend()

                  # Save the plot in the PDF file
                  pdf.savefig()
                  plt.close()


def visualize_yahoo(dataset_raw_dir, outputfile):
    # List all entries in the given raw dataset directory
    dir_list = os.listdir(dataset_raw_dir)

    # Iterate over each entry in the directory list
    for dir_name in dir_list:
        # Construct the full path to the current directory
        full_dir_path = os.path.join(dataset_raw_dir, dir_name)

        # Check if the current entry is a directory and contains 'benchmark' in its name
        if os.path.isdir(full_dir_path) and 'Benchmark' in dir_name:
            print(f"Processing directory: {dir_name}")

            # Create a PDF file for each directory
            os.makedirs(outputfile, exist_ok=True)
            with PdfPages(pdf_filename) as pdf:
                # List all files in the benchmark directory
                for file_name in os.listdir(full_dir_path):
                    if file_name in ["A1Benchmark_all.csv", "A2Benchmark_all.csv", "A3Benchmark_all.csv", "A4Benchmark_all.csv"]:
                        continue

                    # Construct the full path to the current file
                    full_file_path = os.path.join(full_dir_path, file_name)

                    # Check if the current entry is a file
                    if os.path.isfile(full_file_path):

                        # Load the CSV file using pandas
                        df = pd.read_csv(full_file_path)

                        # Rename columns if necessary
                        if 'anomaly' in df.columns:
                            df.rename(columns={'anomaly': 'is_anomaly'}, inplace=True)
                        if 'timestamps' in df.columns:
                            df.rename(columns={'timestamps': 'timestamp'}, inplace=True) 

                        # Create a new plot for each file
                        plt.figure(figsize=(10, 6))
                        sns.lineplot(x='timestamp', y='value', data=df, label='Time Series')

                        # Highlight anomalies with scatter plot
                        anomalies = df[df['is_anomaly'] == 1]
                        sns.scatterplot(x='timestamp', y='value', data=anomalies, color='red', label='Anomalies')

                        # Set plot labels and title
                        plt.xlabel('Timestamp')
                        plt.ylabel('Value')
                        plt.title(f'Time Series with Anomalies - {file_name}')

                        # Save the current plot to the PDF file
                        pdf.savefig()
                        plt.close()






import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import pandas as pd
from datetime import datetime
import seaborn as sns
import pickle


def report_visualizer(pickle_file: str):
    """
    Reads a pickle file containing anomaly detection results (thresholds,
    metrics, anomaly_preds, anomaly_scores) for multiple time-series,
    determines model names and metric names dynamically, then generates
    a multi-row visualization for each time-series:

      Row 0  -> Raw time series + true anomalies
      Row i  -> One row per model (reconstruction error, threshold, predicted anomalies)
      Row M+1 -> Bar charts of all scalar metrics across all models

    Parameters
    ----------
    pickle_file : str
        Path to the pickle file with stored results
        (e.g., "per_file_results.pkl").
    """

    # 1) Load the pickle
    with open(pickle_file, "rb") as f:
        all_results = pickle.load(f)

    print(f"Loaded {len(all_results)} entries from '{pickle_file}'.")

    # 2) Loop over each time-series entry in the pickle
    for entry in all_results:
        train_file = entry["file"]  # e.g. "/path/to/1_train.npy"
        base_name = os.path.basename(train_file)
        result = entry["result"]  # dict with "thresholds" and "metrics"
        thresholds = result.get("thresholds", {})  # e.g. {"AE": val, "VAE": val, ...}
        metrics = result.get("metrics", {})  # e.g. {"AE": {...}, "VAE": {...}, ...}

        # If no metrics or thresholds, skip
        if not metrics or not thresholds:
            print(f"Skipping {base_name}: no metrics or thresholds found.")
            continue

        # 3) Dynamically deduce the model names from the 'metrics' dictionary
        #    e.g. model_names = ["AE", "VAE", "DAE"] (sorted for consistent ordering)
        model_names = sorted(metrics.keys())

        # 4) Collect all possible keys from each model, then exclude arrays
        #    like "anomaly_preds", "anomaly_scores".
        #    Those are used for the reconstruction error rows, not bar charts.
        all_metric_keys = set()
        for m in model_names:
            all_metric_keys.update(metrics[m].keys())
        # Exclude array-type keys
        array_keys = {"anomaly_preds", "anomaly_scores"}
        metric_names = sorted(list(all_metric_keys - array_keys))

        # 5) Build the figure layout
        #    - We have 1 row for the raw time series
        #    - Then 1 row per model for reconstruction error
        #    - Finally 1 row for bar charts of all (scalar) metrics
        num_models = len(model_names)
        num_metrics = len(metric_names)

        # So total rows = num_models + 2
        # columns = max(6, num_metrics) if you want a minimum width,
        # but let's just do columns = num_metrics for the final row.
        nrows = num_models + 2
        ncols = max(1, num_metrics)  # ensure at least 1 col

        fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
        gs = fig.add_gridspec(nrows, ncols)

        # 6) Load the test data and labels for plotting raw time-series
        test_file = train_file.replace("_train.npy", "_test.npy")
        label_file = train_file.replace("_train.npy", "_labels.npy")

        if not os.path.exists(test_file) or not os.path.exists(label_file):
            print(f"Skipping {base_name}: no matching test/label file found.")
            continue

        X_test = np.load(test_file)
        Y_test = np.load(label_file)
        # Flatten if needed
        if len(X_test.shape) > 1:
            X_test = X_test.flatten()
        if len(Y_test.shape) > 1:
            Y_test = Y_test.flatten()

        # 7) Plot row 0 => raw time series + true anomalies
        ax_ts = fig.add_subplot(gs[0, :])  # entire first row
        time_points = np.arange(len(X_test))
        ax_ts.plot(time_points, X_test, color='blue', label='Test Data')

        anom_idx = np.where(Y_test == 1)[0]
        if len(anom_idx) > 0:
            ax_ts.scatter(anom_idx, X_test[anom_idx], color='red', label='True Anomaly')

        ax_ts.set_title(f"Time Series: {base_name}", fontsize=12)
        ax_ts.legend()

        # 8) For each model, we place the reconstruction error plot on row (i+1), entire columns
        for i, m_name in enumerate(model_names):
            row_idx = i + 1  # row 1..(num_models)
            ax_err = fig.add_subplot(gs[row_idx, :])
            model_dict = metrics[m_name]
            # We expect "anomaly_scores" and "anomaly_preds" in model_dict
            anomaly_scores = model_dict.get("anomaly_scores", None)
            anomaly_preds = model_dict.get("anomaly_preds", None)
            threshold = thresholds.get(m_name, None)

            if anomaly_scores is not None and isinstance(anomaly_scores, np.ndarray):
                color_map = {"AE": "blue", "VAE": "green", "DAE": "orange"}
                c = color_map.get(m_name, "black")
                ax_err.plot(anomaly_scores, color=c, label=f"{m_name} Reconstruction Error")
                if threshold is not None:
                    ax_err.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
                if anomaly_preds is not None and isinstance(anomaly_preds, np.ndarray):
                    # highlight predicted anomalies
                    pred_idx = np.where(anomaly_preds == 1)[0]
                    ax_err.scatter(pred_idx, anomaly_scores[pred_idx], color='red', s=20, label='Predicted Anomaly')

            ax_err.set_title(f"{m_name} Reconstruction Error", fontsize=10)
            ax_err.legend()

        # 9) Final row => bar charts for each scalar metric
        bottom_row = num_models + 1
        # We'll create subplots for each metric_name in columns
        # If num_metrics == 0, we skip
        if num_metrics == 0:
            # No scalar metrics to show
            plt.tight_layout()
            plt.show()
            continue

        axes_metrics = []
        for col_idx in range(num_metrics):
            ax = fig.add_subplot(gs[bottom_row, col_idx])
            axes_metrics.append(ax)

        # Now we fill each column with a bar chart comparing the models
        for col_idx, metric_name in enumerate(metric_names):
            ax_bars = axes_metrics[col_idx]

            # Gather metric values from each model
            vals = []
            for m_name in model_names:
                val = metrics[m_name].get(metric_name, np.nan)
                # Only interpret float/int as valid bar values
                if isinstance(val, (float, int)):
                    vals.append(val)
                else:
                    vals.append(np.nan)

            # Make a bar plot: x-axis = model_names, y-values = vals
            color_list = ["skyblue", "lightgreen", "salmon", "lightgray", "yellow"]
            # in case you have more than 3 models
            colors = color_list[:len(model_names)]
            ax_bars.bar(model_names, vals, color=colors)
            ax_bars.set_title(metric_name, fontsize=10)
            if not np.all(np.isnan(vals)):
                ax_bars.set_ylim([0, max(1.0, np.nanmax(vals) * 1.2)])
            # Annotate bars
            for j, v in enumerate(vals):
                if not np.isnan(v):
                    ax_bars.text(j, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)

        plt.tight_layout()
        plt.show()


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






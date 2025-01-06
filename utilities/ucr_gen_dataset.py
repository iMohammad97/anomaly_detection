from utilities.preprocess import preprocess_UCR

dataset_raw_dir = "../UCR_Anomaly_FullData"
dataset_processed_dir = "../UCR"

preprocess_UCR(dataset_raw_dir, dataset_processed_dir)

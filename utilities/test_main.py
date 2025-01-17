from utilities.preprocess import preprocess_UCR
from utilities.visualize import visualizeUCR

if __name__ == '__main__':
    ##################################################
    ############## To call visualizeUCR ##############
    ##################################################
    # raw_input_dir = "../UCR/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData"
    # input_dir = "../UCR/UCR2_preprocessed"
    # output_dir = "report.pdf"
    # output_pdfs_path = "./anomaly_types"
    # anomalies_type_csv_path = "anomaly_types.csv"

    # mode=0, plots all
    # visualizeUCR(input_dir, output_dir)

    # mode=1, plots the list
    # visualizeUCR(input_dir, output_dir, mode=1, list_of_ts=["102", "104"])

    # mode=2, plots an specific type of anomaly
    # visualizeUCR(input_dir, output_dir, mode=2, anomalies_type_csv=anomalies_type_csv_path,
    #              anomaly_type="unusual_pattern")

    # mode=3, plots all types of anomalies, each in a different pdf file
    # visualizeUCR(input_dir, output_dir, mode=3, anomalies_type_csv=anomalies_type_csv_path,
    #              output_pdfs_path=output_pdfs_path)

    ##################################################
    ############ To call preprocess_UCR ##############
    ##################################################
    # input_dir = "../UCR/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData"
    # output_dir = "../UCR/UCR2_preprocessed"
    # output_dir = "../UCR/UCR2_preprocessed_by_source"
    # output_dir = "../UCR/UCR2_preprocessed_by_type"
    # anomalies_type_csv_path = "anomaly_types.csv"

    # To only preprocess all the dataset together
    # preprocess_UCR(input_dir, output_dir)

    # To preprocess the dataset based on TS sources
    # preprocess_UCR(input_dir, output_dir, split_by_source=True)

    # To preprocess the dataset based on TS anomaly type
    # preprocess_UCR(input_dir, output_dir, split_by_anomaly_type=True, anomalies_type_csv=anomalies_type_csv_path)

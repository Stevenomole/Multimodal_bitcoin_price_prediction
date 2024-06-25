import os
import shutil
from data_ingestion.data_cleaning import handle_missing, load_data
from data_ingestion.data_preparation import generate_candlestick_charts, transform_ts_data, split_images
from models.train import train_models
from models.predict import load_and_predict

def main():
    # Data Cleaning
    file_path = "data/raw/Boruta_onchain_data.csv"
    print(f"Loading data from {file_path}.")
    data = load_data(file_path)

    if data is not None:
        print("Data loaded. Starting data cleaning.")
        cleaned_data = handle_missing(data)

        if cleaned_data is not None:
            output_path = "data/preprocessed/preprocessed.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cleaned_data.to_csv(output_path, index=False)
            print(f"Data cleaning completed and saved to {output_path}.")
        else:
            print("Data cleaning failed.")
    else:
        print("Data loading failed.")
        return

    # Data Preparation
    data = load_data("data/preprocessed/preprocessed.csv")
    if data is not None:
        generate_candlestick_charts(data, chart_length=14)
        
        transform_ts_data(data)
        
        split_images()

    # Model Training
    train_models()

    # Prediction and Comparison
    load_and_predict()

if __name__ == "__main__":
    main()

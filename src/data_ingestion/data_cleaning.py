import pandas as pd
import os
from utils.other_functions import load_data

def handle_missing(data):
    """
    Handles missing values, ensures consistency in data types. Return preprocessed data as dataframe.
    """
    try:
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        print("Handling missing data completed.")
        return data
    except Exception as e:
        print(f"Error in preprocessing data: {e}.")
        return None
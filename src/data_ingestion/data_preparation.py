import os
import talib
import pickle
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mplfinance.original_flavor import candlestick_ohlc
from utils.other_functions import load_data

def generate_candlestick_charts(data, chart_length=14):
    """
    This function generates charts from the price data and save them in data/charts.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data[['timestamp', 'price-ohlc-usd-o', 'price-ohlc-usd-h', 'price-ohlc-usd-l', 'price-ohlc-usd-c']].dropna()

    # Calculate technical indicators
    data['SMA'] = talib.SMA(data['price-ohlc-usd-c'], timeperiod=20)
    data['EMA'] = talib.EMA(data['price-ohlc-usd-c'], timeperiod=20)
    data['Upper_BB'], _, data['Lower_BB'] = talib.BBANDS(data['price-ohlc-usd-c'], timeperiod=20)
    
    # Calculate MACD and Signal lines
    data['MACD'], data['Signal'], data['MACD_Hist'] = talib.MACD(data['price-ohlc-usd-c'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    data[['MACD', 'Signal', 'MACD_Hist']] = data[['MACD', 'Signal', 'MACD_Hist']].fillna(0)

    targets = []
      
    for i in range(20, len(data) - chart_length):  # Ensure there is enough data for all indicators
        weekly_data = data.iloc[i:i+chart_length].copy()
    
        ohlc_data = weekly_data[['timestamp', 'price-ohlc-usd-o', 'price-ohlc-usd-h', 'price-ohlc-usd-l', 'price-ohlc-usd-c']].copy()
        ohlc_data['timestamp'] = mdates.date2num(ohlc_data['timestamp'].to_list())
        ohlc_data = ohlc_data[['timestamp', 'price-ohlc-usd-o', 'price-ohlc-usd-h', 'price-ohlc-usd-l', 'price-ohlc-usd-c']].values

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.grid(False)

        # Set y-axis limits based on the range of data to improve scaling
        ax1.set_ylim([weekly_data['price-ohlc-usd-l'].min() * 0.95, weekly_data['price-ohlc-usd-h'].max() * 1.05])

        candlestick_ohlc(ax1, ohlc_data, width=0.3, colorup='g', colordown='r')  # Making candles slimmer

        # Plot SMA and EMA
        ax1.plot(weekly_data['timestamp'], weekly_data['SMA'], color='blue')
        ax1.plot(weekly_data['timestamp'], weekly_data['EMA'], color='orange')

        # Plot Bollinger Bands and fill area between them
        ax1.plot(weekly_data['timestamp'], weekly_data['Upper_BB'], color='grey')
        ax1.plot(weekly_data['timestamp'], weekly_data['Lower_BB'], color='grey')
        ax1.fill_between(weekly_data['timestamp'], weekly_data['Upper_BB'], weekly_data['Lower_BB'], color='lightblue', alpha=0.3)

        # Plot MACD Histogram with conditional colors
        colors = ['green' if val >= 0 else 'red' for val in weekly_data['MACD_Hist']]
        ax2.bar(weekly_data['timestamp'], weekly_data['MACD_Hist'], color=colors, alpha=0.3)

        # Remove legends, titles, and labels
        ax1.set_title('')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        ax2.set_title('')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        ax1.axis('off')
        ax2.axis('off')
        
        plt.tight_layout()

        chart_dir = "data/charts/all_charts"
        os.makedirs(chart_dir, exist_ok=True)
        chart_filename = weekly_data.iloc[-1]['timestamp'].strftime('%Y-%m-%d') + '.png'
        plt.savefig(os.path.join(chart_dir, chart_filename))
        plt.close(fig)

    print("Generating price charts completed.")

def create_sequences(data, targets, timesteps):
    X = []
    y = []
    timestamps = []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:i + timesteps, 1:])  # Exclude the timestamp column from the sequences
        y.append(targets[i + timesteps - 1])  # Target corresponding to the last observation in the sequence
        timestamps.append(data[i + timesteps - 1, 0])  # Keep the timestamp of the last observation in the sequence
    return np.array(X), np.array(y), np.array(timestamps)

def transform_ts_data(data):
    """
    Transforms the time series data and saves them as pickle files.
    """
    # Calculate target for time series data
    price = pd.DataFrame()
    price['today'] = data['price-ohlc-usd-c']
    price['next day'] = price['today'].shift(-1)
    data['Target'] = (price['next day'] > price['today']).astype(int)

    # Create X and y
    X = data.drop(['Target'], axis=1)
    y = data[['timestamp', 'Target']]

    X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Separate the timestamp column
    X_train_ts_timestamp = X_train_ts['timestamp']
    X_test_ts_timestamp = X_test_ts['timestamp']

    # Drop the timestamp column for scaling
    X_train_ts_data = X_train_ts.drop(columns=['timestamp'])
    X_test_ts_data = X_test_ts.drop(columns=['timestamp'])

    # Scale the input data
    scaler = StandardScaler()
    X_train_ts_scaled = scaler.fit_transform(X_train_ts_data)
    X_test_ts_scaled = scaler.transform(X_test_ts_data)

    # Convert back to DataFrame to reattach timestamps
    X_train_ts_scaled = pd.DataFrame(X_train_ts_scaled, columns=X_train_ts_data.columns)
    X_train_ts_scaled['timestamp'] = X_train_ts_timestamp.values
    X_train_ts_scaled = X_train_ts_scaled[['timestamp'] + [col for col in X_train_ts_scaled.columns if col != 'timestamp']]
    
    X_test_ts_scaled = pd.DataFrame(X_test_ts_scaled, columns=X_test_ts_data.columns)
    X_test_ts_scaled['timestamp'] = X_test_ts_timestamp.values
    X_test_ts_scaled = X_test_ts_scaled[['timestamp'] + [col for col in X_test_ts_scaled.columns if col != 'timestamp']]

    # Convert DataFrame to NumPy array with dtype=object to maintain date format
    X_train_ts_scaled = X_train_ts_scaled.to_numpy(dtype=object)
    X_test_ts_scaled = X_test_ts_scaled.to_numpy(dtype=object)

    timesteps = 5
    X_train_ts_reshaped, y_train_ts, train_timestamps = create_sequences(X_train_ts_scaled, y_train_ts.values, timesteps)
    X_test_ts_reshaped, y_test_ts, test_timestamps = create_sequences(X_test_ts_scaled, y_test_ts.values, timesteps)

    # Save data as pickle files
    os.makedirs('data/timeseries', exist_ok=True)
    with open('data/timeseries/X_train.pkl', 'wb') as f:
        pickle.dump(X_train_ts_reshaped, f)
    with open('data/timeseries/y_train.pkl', 'wb') as f:
        pickle.dump(y_train_ts, f)
    with open('data/timeseries/train_timestamps.pkl', 'wb') as f:
        pickle.dump(train_timestamps, f)

    with open('data/timeseries/X_test.pkl', 'wb') as f:
        pickle.dump(X_test_ts_reshaped, f)
    with open('data/timeseries/y_test.pkl', 'wb') as f:
        pickle.dump(y_test_ts, f)
    with open('data/timeseries/test_timestamps.pkl', 'wb') as f:
        pickle.dump(test_timestamps, f)

    print("Timeseries data transformed and saved successfully.")


def split_images():
    """ 
    This function splits the images and their targets into training and testing using the timeseries corresponding dates.
    """
    # Load the train and test timestamps from pickle files
    with open('data/timeseries/train_timestamps.pkl', 'rb') as f:
        train_timestamps = pickle.load(f)
    with open('data/timeseries/test_timestamps.pkl', 'rb') as f:
        test_timestamps = pickle.load(f)

    # Load the train and test target values from pickle files
    with open('data/timeseries/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/timeseries/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    # Convert timestamps to the format used in the filenames
    train_dates = [pd.to_datetime(ts).strftime('%Y-%m-%d') for ts in train_timestamps]
    test_dates = [pd.to_datetime(ts).strftime('%Y-%m-%d') for ts in test_timestamps]

    # Define source and destination directories
    source_dir = 'data/charts/all_charts'
    train_dir = 'data/charts/train_charts'
    test_dir = 'data/charts/test_charts'

    # Create the train and test directories if they do not already exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move images to their respective directories based on the dates
    for img_file in os.listdir(source_dir):
        if img_file.endswith('.png'):
            date_str = img_file.replace('.png', '')
            if date_str in train_dates:
                shutil.move(os.path.join(source_dir, img_file), os.path.join(train_dir, img_file))
            elif date_str in test_dates:
                shutil.move(os.path.join(source_dir, img_file), os.path.join(test_dir, img_file))

    # Save the target values for the image train and test sets
    with open('data/charts/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('data/charts/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

    print("Images and targets successfully separated and saved.")



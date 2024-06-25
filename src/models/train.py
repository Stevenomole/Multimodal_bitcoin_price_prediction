import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model, save_model # type: ignore
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from utils.other_functions import reset_random_seeds
from models.image_classification_model import create_cnn_model
from models.timeseries_model import create_cnn_lstm_model
from utils.other_functions import convert_images_to_arrays

def train_models():
    # Load the input and target values
    with open('data/timeseries/X_train.pkl', 'rb') as f:
        X_train_ts = pickle.load(f)

    with open('data/timeseries/y_train.pkl', 'rb') as f:
        y_train_ts = pickle.load(f)[:,1]

    with open('data/charts/y_train.pkl', 'rb') as f:
        y_train_img = pickle.load(f)[:,1]

    image_df = convert_images_to_arrays("data/charts/train_charts")
        
    X_train_img = np.stack(image_df['image'].values)

    # Ensure that all data have same length and are properly aligned
    min_length = min(len(X_train_img), len(X_train_ts))
    X_train_ts = X_train_ts[-min_length:]
    X_train_img = X_train_img[-min_length:]
    y_train_ts = y_train_ts[-min_length:]
    y_train_img = y_train_img[-min_length:]

    # Convert the target values to type int32
    X_train_ts = X_train_ts.astype('float')
    X_train_img = X_train_img.astype('float')
    y_train_ts = y_train_ts.astype('int32')
    y_train_img = y_train_img.astype('int32')
    
    # Train CNN model for image data
    reset_random_seeds()
    img_inputs = Input(shape=(64, 64, 3))
    img_model = create_cnn_model(img_inputs)
    early_stopping_img = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.001)
    img_model.fit(X_train_img, y_train_img, epochs=1000, batch_size=32, validation_split=0.1, callbacks=[early_stopping_img, reduce_lr])
    img_outputs = img_model.layers[-3].output

    # Extract features from CNN model
    img_feature_extractor = Model(inputs=img_inputs, outputs=img_outputs)
    train_img_features = img_feature_extractor.predict(X_train_img, batch_size=32)
    
    print("Image classification model successfully trained.")

    # Train CNN-LSTM model for time series data
    reset_random_seeds()
    timesteps = 5
    ts_inputs = Input(shape=(timesteps, X_train_ts.shape[2]))
    ts_model = create_cnn_lstm_model(ts_inputs)
    early_stopping_ts = EarlyStopping(monitor='val_loss', patience=100)
    ts_model.fit(X_train_ts, y_train_ts, epochs=1000, batch_size=50, validation_split=0.1, callbacks=[early_stopping_ts])
    ts_outputs = ts_model.layers[-2].output

    # Extract features from CNN-LSTM model
    ts_feature_extractor = Model(inputs=ts_inputs, outputs=ts_outputs)
    train_ts_features = ts_feature_extractor.predict(X_train_ts, batch_size=50)

    print("Timeseries classification model successfully trained.")

    # Combine features
    X_train_combined = np.concatenate((train_img_features, train_ts_features), axis=1)

    # Train final model
    reset_random_seeds()
    final_model = RandomForestClassifier(n_estimators=200, random_state=1)
    final_model.fit(X_train_combined, y_train_ts)

    print("Final model successfully trained.")

    # Create directory to save models
    output_dir = 'src/models_output'
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    save_model(img_feature_extractor, os.path.join(output_dir, 'img_feature_extractor.h5'))
    save_model(ts_feature_extractor, os.path.join(output_dir, 'ts_feature_extractor.h5'))
    img_model.save(os.path.join(output_dir, 'img_model.h5'))
    ts_model.save(os.path.join(output_dir, 'ts_model.h5'))

    with open(os.path.join(output_dir, 'final_model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)
   
    print("All models successfully saved.")

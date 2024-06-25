import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.other_functions import convert_images_to_arrays

def load_and_predict():
    # Load the feature extractors
    img_feature_extractor = load_model('src/models_output/img_feature_extractor.h5')
    ts_feature_extractor = load_model('src/models_output/ts_feature_extractor.h5')

    # Load the final model
    with open('src/models_output/final_model.pkl', 'rb') as f:
        final_model = pickle.load(f)

    # Load the individual image classification model
    img_model = load_model('src/models_output/img_model.h5')

    # Load the individual time series model
    ts_model = load_model('src/models_output/ts_model.h5')

    # Load the test data for time series
    with open('data/timeseries/X_test.pkl', 'rb') as f:
        X_test_ts = pickle.load(f)

    with open('data/timeseries/y_test.pkl', 'rb') as f:
        y_test_ts = pickle.load(f)[:, 1] 

    # Convert images to arrays and load target values for image data
    image_df = convert_images_to_arrays("data/charts/test_charts")
    X_test_img = np.stack(image_df['image'].values)

    with open('data/charts/y_test.pkl', 'rb') as f:
        y_test_img = pickle.load(f)[:, 1]  

    # Ensure that all data have same length and are properly aligned
    min_length = min(len(X_test_img), len(X_test_ts))
    X_test_ts = X_test_ts[-min_length:].astype('float32')
    X_test_img = X_test_img[-min_length:].astype('float32')
    y_test_ts = y_test_ts[-min_length:].astype('int32')
    y_test_img = y_test_img[-min_length:].astype('int32')

    # Extract features from both models
    test_cnn_features = img_feature_extractor.predict(X_test_img, batch_size=32)
    test_lstm_features = ts_feature_extractor.predict(X_test_ts, batch_size=50)

    # Combine features
    X_test_combined = np.concatenate((test_cnn_features, test_lstm_features), axis=1)

    # Predictions for the test data using the combined model
    y_test_pred_combined = final_model.predict(X_test_combined)

    # Predictions for the test data using individual models
    y_test_pred_img = img_model.predict(X_test_img)
    y_test_pred_ts = ts_model.predict(X_test_ts)

    # Convert probabilities to binary class labels
    y_test_pred_combined = (y_test_pred_combined >= 0.5).astype(int)
    y_test_pred_img = (y_test_pred_img >= 0.5).astype(int)
    y_test_pred_ts = (y_test_pred_ts >= 0.5).astype(int)

    # Calculate metrics for combined model
    combined_accuracy = accuracy_score(y_test_ts, y_test_pred_combined)
    combined_precision = precision_score(y_test_ts, y_test_pred_combined)
    combined_recall = recall_score(y_test_ts, y_test_pred_combined)
    combined_f1 = f1_score(y_test_ts, y_test_pred_combined)

    # Calculate metrics for image model
    img_accuracy = accuracy_score(y_test_img, y_test_pred_img)
    img_precision = precision_score(y_test_img, y_test_pred_img)
    img_recall = recall_score(y_test_img, y_test_pred_img)
    img_f1 = f1_score(y_test_img, y_test_pred_img)

    # Calculate metrics for time series model
    ts_accuracy = accuracy_score(y_test_ts, y_test_pred_ts)
    ts_precision = precision_score(y_test_ts, y_test_pred_ts)
    ts_recall = recall_score(y_test_ts, y_test_pred_ts)
    ts_f1 = f1_score(y_test_ts, y_test_pred_ts)

    # Print the results for combined model
    print("Evaluating Combined Model...")
    print(f'Combined Model Test Accuracy: {combined_accuracy:.4f}')
    print(f'Combined Model Test Precision: {combined_precision:.4f}')
    print(f'Combined Model Test Recall: {combined_recall:.4f}')
    print(f'Combined Model Test F1 Score: {combined_f1:.4f}')
    print()

    # Print the results for image model
    print("Evaluating Image Model...")
    print(f'Image Model Test Accuracy: {img_accuracy:.4f}')
    print(f'Image Model Test Precision: {img_precision:.4f}')
    print(f'Image Model Test Recall: {img_recall:.4f}')
    print(f'Image Model Test F1 Score: {img_f1:.4f}')
    print()

    # Print the results for time series model
    print("Evaluating Time Series Model...")
    print(f'Time Series Model Test Accuracy: {ts_accuracy:.4f}')
    print(f'Time Series Model Test Precision: {ts_precision:.4f}')
    print(f'Time Series Model Test Recall: {ts_recall:.4f}')
    print(f'Time Series Model Test F1 Score: {ts_f1:.4f}')
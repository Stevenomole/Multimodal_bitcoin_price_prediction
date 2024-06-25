import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, AveragePooling1D, LSTM, Dense, Dropout, BatchNormalization, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def create_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(input_shape)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=1))
    model.add(Dropout(0.5))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(units=80))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
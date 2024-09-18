import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np

#%%

# Define LSTM model
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape_lstm = (None, 10, 32)  # Example input shape for LSTM
num_classes = 10  # Example number of classes
lstm_model = create_lstm_model(input_shape_lstm, num_classes)
lstm_model.summary()
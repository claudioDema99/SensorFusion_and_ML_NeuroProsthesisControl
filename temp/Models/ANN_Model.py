import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np

#%%

# Define simple ANN model
# 20 hidden neurons, good for load rec. from bicept
def create_ann_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape_ann = (28, 28)  # Example input shape for ANN
num_classes = 10  # Example number of classes
ann_model = create_ann_model(input_shape_ann, num_classes)
ann_model.summary()
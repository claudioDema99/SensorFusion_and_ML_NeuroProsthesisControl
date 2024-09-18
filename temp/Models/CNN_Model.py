#%%
# Reshape the data for fit the CNN keras model
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
import numpy as np

#%%


def reshape_data_for_cnn(data, num_channels, num_samples_per_movement, num_movements):
    reshaped_data = np.empty((num_channels, num_samples_per_movement * num_movements))
    for i in range(num_movements):
        start_idx = i * num_samples_per_movement
        end_idx = start_idx + num_samples_per_movement
        reshaped_data[:, start_idx:end_idx] = data[:, :, i]
    return reshaped_data

# Define CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Simple CNN for Reinforcement Learning
class Simple1DCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=6, kernel_size=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Usage example
input_shape = (None, 11, 6)  # Input shape: (batch_size, sequence_length, num_channels)
num_classes = 10  # Number of output classes
model = Simple1DCNN(num_classes)

# Other CNN example:
class CNNForEMG(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(CNNForEMG, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Usage example
input_shape = (window_length, num_channels, 1)  # Input shape: (time_step, num_channels, 1)
num_classes = 10  # Number of output classes
model = CNNForEMG(input_shape, num_classes)

#%%

# Load the signal data
signal_data_path = 'C:/Users/claud/Desktop/LocoD-Offline/SavedData/Dataset/data.npy'
signal_data = np.load(signal_data_path)
dataset = reshape_data_for_cnn(signal_data, signal_data.shape[0], signal_data.shape[1], signal_data.shape[2])
# Split the data into training and validation sets
labels = np.array([0, 1, 2, 3] * 25)
X_train, X_val, y_train, y_val = train_test_split(dataset.T, labels, test_size=0.2, random_state=42)
# model
input_shape_cnn = (signal_data.shape[0], signal_data.shape[1], 1)
num_classes = 4  # Example number of classes
cnn_model = create_cnn_model(input_shape_cnn, num_classes)

#%%

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
# Evaluate the model
loss, accuracy = cnn_model.evaluate(X_val, y_val)
print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')

#%%

num_classes = 4  # Number of output classes

# Define separate input layers for each type of sensor data
input_emg = layers.Input(shape=([80, 29]))
input_imu = layers.Input(shape=([80, 2]))

# EMG branch
x_emg = layers.Conv1D(32, 3, activation='relu')(input_emg)
x_emg = layers.MaxPooling1D(2)(x_emg)
# Additional layers for EMG data processing

# IMU branch
x_imu = layers.Conv1D(32, 3, activation='relu')(input_imu)
x_imu = layers.MaxPooling1D(2)(x_imu)
# Additional layers for IMU data processing

# Merge branches
merged = layers.concatenate([x_emg, x_imu])

# Dense layers for further processing
x = layers.Flatten()(merged)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(num_classes, activation='softmax')(x)

# Define the model
model = models.Model(inputs=[input_emg, input_imu], outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%

# Train the model
model.fit([x_train_emg, x_train_imu], y_train, epochs=10, batch_size=32, validation_data=([x_test_emg, x_test_imu], y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate([x_test_emg, x_test_imu], y_test)
print('Test accuracy:', test_acc)

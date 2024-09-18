import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np

#%%

# Multimodal Fusion CNN
class FirstStreamCNN(tf.keras.Model):
    def __init__(self):
        super(FirstStreamCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=23, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=0.001)
        self.relu = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=15, strides=15)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        
        self.conv2 = tf.keras.layers.Conv1D(filters=96, kernel_size=12, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=0.001)
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        
        self.conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=11, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=0.001)

    def call(self, inputs):
        x = self.dropout1(self.pool1(self.relu(self.bn1(self.conv1(inputs)))))
        x = self.dropout2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class SecondStreamCNN(tf.keras.Model):
    def __init__(self):
        super(SecondStreamCNN, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same')
        self.bn = tf.keras.layers.BatchNormalization(epsilon=0.001)
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs):
        x = self.dropout(self.pool(self.relu(self.bn(self.conv(inputs)))))
        return x

class MultimodalFusionCNN(tf.keras.Model):
    def __init__(self):
        super(MultimodalFusionCNN, self).__init__()
        self.first_stream = FirstStreamCNN()
        self.second_stream = SecondStreamCNN()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, emg, imu):
        emg_features = self.first_stream(emg)
        imu_features = self.second_stream(imu)
        fused_features = tf.concat([emg_features, imu_features], axis=1)
        output = self.fc(fused_features)
        return output

# Usage example
num_classes = 10  # Number of output classes
model = MultimodalFusionCNN()


# RCNN
class RCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        self.sequence_input = tf.keras.layers.InputLayer(input_shape=(None, input_dim))
        self.sequence_folding = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))
        
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

        self.conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        self.pool3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

        self.conv4 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.ReLU()
        self.pool4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

        self.conv5 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.relu5 = tf.keras.layers.ReLU()

        self.sequence_unfolding = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512))

        self.flatten = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(128)
        self.fc = tf.keras.layers.Dense(num_classes, activation='relu')
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = self.sequence_input(inputs)
        x = self.sequence_folding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.sequence_unfolding(x)

        x = self.flatten(x)
        x = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# Usage example
input_dim = 50  # Dimensionality of input features
num_classes = 10  # Number of output classes
model = RCNN(num_classes)


# CNN + ANN

# Define the architecture of the neural network
def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # CNN layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu')) # First hidden layer
    model.add(layers.Dense(64, activation='relu')) # Second hidden layer
    model.add(layers.Dense(32, activation='relu')) # Third hidden layer
    model.add(layers.Dense(16, activation='relu')) # Fourth hidden layer
    model.add(layers.Dense(8, activation='relu')) # Fifth hidden layer
    model.add(layers.Dense(4, activation='relu')) # Sixth hidden layer
    model.add(layers.Dense(num_classes, activation='softmax')) # Output layer
    return model

# Define your input shape and number of classes
input_shape = (29, 1000, 1) # Example input shape of dimension 29x1000
num_classes = 4 # Example number of classes for classification task

# Create the model
model = create_model(input_shape, num_classes)

windows = np.array(windows)

x_train, x_test, y_train, y_test = train_test_split(windows, labels, test_size=0.2, random_state=42)

param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')

# Fit the grid search
grid_result = grid.fit(x_train, y_train)

# Get the best hyperparameters and evaluate the model
best_params = grid_result.best_params_
best_model = grid_result.best_estimator_

# Train the best model with the best hyperparameters
best_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

# Evaluate the model
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best Hyperparameters:", best_params)
print("Test Accuracy:", accuracy)


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# CNN

num_classes = 4  # Number of output classes

# Define separate input layers for each type of sensor data
input_emg = layers.Input(shape=(29, 1000))  # Adjusted shape: (num_samples, num_channels)
input_imu = layers.Input(shape=(2, 1000))   # Adjusted shape: (num_samples, num_channels)

# Permute the dimensions to match the Conv1D input format
permuted_input_emg = layers.Permute((2, 1))(input_emg)
permuted_input_imu = layers.Permute((2, 1))(input_imu)

# EMG branch
x_emg = layers.Conv1D(512, 3, activation='relu')(permuted_input_emg)
x_emg = layers.MaxPooling1D(2)(x_emg)
x_emg = layers.Conv1D(256, 3, activation='relu')(x_emg)  # Additional convolutional layer
x_emg = layers.MaxPooling1D(2)(x_emg)    
x_emg = layers.Conv1D(128, 3, activation='relu')(x_emg)  # Additional convolutional layer
x_emg = layers.MaxPooling1D(2)(x_emg)                    # Additional pooling layer
x_emg = layers.Dropout(0.25)(x_emg)                      # Dropout layer
x_emg = layers.BatchNormalization()(x_emg)               # Batch normalization layer

# IMU branch
x_imu = layers.Conv1D(512, 3, activation='relu')(permuted_input_imu)
x_imu = layers.MaxPooling1D(2)(x_imu)
x_imu = layers.Conv1D(256, 3, activation='relu')(x_imu)  # Additional convolutional layer
x_imu = layers.MaxPooling1D(2)(x_imu) 
x_imu = layers.Conv1D(128, 3, activation='relu')(x_imu)  # Additional convolutional layer
x_imu = layers.MaxPooling1D(2)(x_imu)                    # Additional pooling layer
x_imu = layers.Dropout(0.25)(x_imu)                      # Dropout layer
x_imu = layers.BatchNormalization()(x_imu)               # Batch normalization layer

# Merge branches
merged = layers.concatenate([x_emg, x_imu])

# Dense layers for further processing
x = layers.Flatten()(merged)
x = layers.Dense(128, activation='relu')(x)  # Additional dense layer
x = layers.Dropout(0.5)(x)                   # Dropout layer
x = layers.BatchNormalization()(x)           # Batch normalization layer
output = layers.Dense(num_classes, activation='softmax')(x)

# Define the model
model = models.Model(inputs=[input_emg, input_imu], outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

windows = np.array(windows)
angles_windowed = np.array(angles_windowed)
labels = np.array(labels)

# Check the number of elements in each list
num_samples_windows = len(windows)
num_samples_angles = len(angles_windowed)
num_samples_label = len(labels)

print(num_samples_windows, num_samples_angles, num_samples_label)

# Ensure all lists have the same number of elements
assert num_samples_windows == num_samples_angles == num_samples_label, "Number of samples in input lists are not consistent"


x_train_emg, x_test_emg, x_train_imu, x_test_imu, y_train, y_test = train_test_split(windows, angles_windowed, labels, test_size=0.2, random_state=42)

# Train the model
model.fit([x_train_emg, x_train_imu], y_train, epochs=50, batch_size=32, validation_data=([x_test_emg, x_test_imu], y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate([x_test_emg, x_test_imu], y_test)
print('Test accuracy:', test_acc)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Make predictions on the test data
y_pred = model.predict([x_test_emg, x_test_imu])

# Convert the predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Assuming y_true contains the true labels and y_pred contains the predicted labels
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# LSTM

# Define LSTM model
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(100, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape_lstm = (29, 7)  # Example input shape for LSTM
num_classes = 4  # Example number of classes
lstm_model = create_lstm_model(input_shape_lstm, num_classes)

# Step 2: Compile the LSTM model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Concatenate arr1 and arr2 vertically
x_dataset = feature_vector
'''
for i, val in enumerate(windows):
    x_dataset.append(np.vstack((val, angles_windowed[i])))
'''

x_dataset = np.array(x_dataset)
labels_ = np.vstack((labels, labels))
labels = np.vstack((labels_,labels_))
labels = np.array(tf.reshape(labels, (-1, 1)))

# Check the number of elements in each list
num_dataset = len(x_dataset)
num_label = len(labels.T)

print(num_dataset, num_label)

# Ensure all lists have the same number of elements
assert num_dataset == num_label, "Number of samples in input lists are not consistent"
    
x_train, x_test, y_train, y_test = train_test_split(x_dataset, labels, test_size=0.2, random_state=42)

# Train the model
lstm_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = lstm_model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
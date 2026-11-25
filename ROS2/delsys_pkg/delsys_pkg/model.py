import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import wandb

num_emg_channels = 9

#%% LSTM network

class MyMultimodalNetworkLSTM(nn.Module):
    def __init__(self, input_shape_emg, input_shape_imu, num_classes, hidden_sizes_emg, hidden_sizes_imu, dropout_rate, raw_imu=None, squeeze=None):
        super(MyMultimodalNetworkLSTM, self).__init__()
        self.emg_input_shape = input_shape_emg
        self.imu_input_shape = input_shape_imu
        self.num_classes = num_classes
        self.hidden_sizes_emg = hidden_sizes_emg
        self.hidden_sizes_imu = hidden_sizes_imu
        self.dropout_rate = dropout_rate
        if squeeze is not None:
            self.squeeze = True
        else:
            self.squeeze = False

        # EMG pathway using LSTM
        self.emg_lstm_layers = nn.ModuleList()
        if raw_imu == True:
            emg_input_size = self.emg_input_shape
        else:
            emg_input_size = self.emg_input_shape[1]  # Input features for LSTM
        for emg_hidden_size in self.hidden_sizes_emg:
            self.emg_lstm_layers.append(nn.LSTM(emg_input_size, emg_hidden_size, batch_first=True))
            emg_input_size = emg_hidden_size

        # IMU pathway using LSTM
        self.imu_lstm_layers = nn.ModuleList()
        imu_input_size = self.imu_input_shape  # Input features for LSTM
        for imu_hidden_size in self.hidden_sizes_imu:
            self.imu_lstm_layers.append(nn.LSTM(imu_input_size, imu_hidden_size, batch_first=True))
            imu_input_size = imu_hidden_size

        # Fully connected layer for concatenation
        fc_input_size = self.hidden_sizes_emg[-1] + self.hidden_sizes_imu[-1]
        self.fc_concat = nn.Linear(fc_input_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emg, imu):
        # Check if input tensors have the correct shape
        batch_size = emg.size(0)
        if not self.squeeze:
            emg = emg.reshape(batch_size,1,num_emg_channels*4)

        # EMG pathway
        for emg_lstm in self.emg_lstm_layers:
            emg, _ = emg_lstm(emg)
        emg = emg[:, -1, :]  # Output from the last time step

        # IMU pathway
        if len(imu.size()) == 2:
            imu = imu.unsqueeze(1)  # Add a sequence dimension: (batch_size, 1, features)
        for imu_lstm in self.imu_lstm_layers:
            imu, _ = imu_lstm(imu)
        imu = imu[:, -1, :] # Output from the last time step

        # Concatenate EMG and IMU outputs
        concat_out = torch.cat((emg, imu), dim=1)

        # Fully connected layer
        output = self.fc_concat(concat_out)
        # output = self.softmax(output)

        return output

def train_lstm(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        exact_matches = 0
        
        for emg, imu, label in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(emg, imu)

            # Compute loss
            loss = criterion(outputs, label)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()

            # Exact match ratio calculation
            exact_matches += (predicted == labels_idx).sum().item() == label.size(0)

        # Compute average loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        emr = exact_matches / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, EMR: {emr}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

    return train_loss, train_accuracy

def evaluate_lstm(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for emg, imu, label in val_loader:
            outputs = model(emg, imu)
            loss = criterion(outputs, label)
            running_loss += loss.item()

            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()
            y_true.extend(labels_idx.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total

    return val_loss, val_accuracy, y_true, y_pred

def inference_lstm(model, emg_input, imu_input):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(emg_input, imu_input)
        _, predicted = torch.max(output, 1)
    return predicted

class MyNetworkLSTM(nn.Module):
    def __init__(self, input_shape_emg, num_classes, hidden_sizes_emg, dropout_rate):
        super(MyNetworkLSTM, self).__init__()
        self.emg_input_shape = input_shape_emg
        self.num_classes = num_classes
        self.hidden_sizes_emg = hidden_sizes_emg
        self.dropout_rate = dropout_rate

        # Ensure input_shape_emg is a tuple
        if isinstance(self.emg_input_shape, list):
            self.emg_input_shape = tuple(self.emg_input_shape)
        
        # EMG pathway using LSTM
        self.emg_lstm_layers = nn.ModuleList()
        emg_input_size = self.emg_input_shape[1]  # Input features for LSTM
        for emg_hidden_size in self.hidden_sizes_emg:
            self.emg_lstm_layers.append(nn.LSTM(emg_input_size, emg_hidden_size, batch_first=True))
            emg_input_size = emg_hidden_size

        # Fully connected layer
        fc_input_size = self.hidden_sizes_emg[-1]
        self.fc_concat = nn.Linear(fc_input_size, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emg): 
        # EMG pathway
        for emg_lstm in self.emg_lstm_layers:
            emg, _ = emg_lstm(emg)        
        emg = emg[:, -1, :]  # Output from the last time step
        # Fully connected layer
        output = self.fc_concat(emg)
        output = self.dropout(output)
        # output = self.softmax(output)
        return output

def train_EMG_lstm(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        exact_matches = 0
        
        for emg, label in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(emg)

            # Compute loss
            loss = criterion(outputs, label)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()

            # Exact match ratio calculation
            exact_matches += (predicted == labels_idx).sum().item() == label.size(0)

        # Compute average loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        emr = exact_matches / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, EMR: {emr}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

    return train_loss, train_accuracy

def evaluate_EMG_lstm(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for emg, label in val_loader:
            outputs = model(emg)
            loss = criterion(outputs, label)
            running_loss += loss.item()

            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()
            y_true.extend(labels_idx.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total

    return val_loss, val_accuracy, y_true, y_pred

#%% CNN network

class MyMultimodalNetworkCNN(nn.Module):
    def __init__(self, input_shape_emg, input_shape_imu, num_classes, hidden_sizes_emg, hidden_sizes_imu, dropout_rate, raw_imu=None):
        super(MyMultimodalNetworkCNN, self).__init__()
        self.emg_input_shape = input_shape_emg
        self.imu_input_shape = input_shape_imu
        self.num_classes = num_classes
        self.hidden_sizes_emg = hidden_sizes_emg
        self.hidden_sizes_imu = hidden_sizes_imu
        self.dropout_rate = dropout_rate

        # EMG pathway using CNN
        self.emg_conv_layers = nn.ModuleList()
        self.emg_pool = nn.MaxPool1d(kernel_size=2)
        emg_input_channels = 1  # Since EMG input is (batch, 11, 4), treat it as 1-channel signal
        if raw_imu == True:
            emg_output_size = self.emg_input_shape
        else:
            emg_output_size = self.emg_input_shape[0] * self.emg_input_shape[1]
        for emg_hidden_size in self.hidden_sizes_emg:
            self.emg_conv_layers.append(nn.Conv1d(emg_input_channels, emg_hidden_size, kernel_size=3, padding=1))
            emg_input_channels = emg_hidden_size
            emg_output_size = emg_output_size // 2  # Assuming MaxPooling reduces by half each time

        # IMU pathway using CNN
        self.imu_conv_layers = nn.ModuleList()
        self.imu_pool = nn.MaxPool1d(kernel_size=2)
        imu_input_channels = 1  # Since IMU input is (batch, 1, 9), treat it as 1-channel signal
        imu_output_size = self.imu_input_shape
        for imu_hidden_size in self.hidden_sizes_imu:
            self.imu_conv_layers.append(nn.Conv1d(imu_input_channels, imu_hidden_size, kernel_size=3, padding=1))
            imu_input_channels = imu_hidden_size
            imu_output_size = imu_output_size // 2  # Assuming MaxPooling reduces by half each time

        # Fully connected layer for concatenation
        fc_input_size = (emg_output_size + imu_output_size) * self.hidden_sizes_emg[-1]
        self.fc_concat = nn.Linear(fc_input_size, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emg, imu):
        # EMG pathway
        emg = emg.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 11, 4)
        emg = emg.view(emg.size(0), 1, -1)  # Reshape to (batch_size, 1, 44)
        for emg_conv in self.emg_conv_layers:
            emg = self.emg_pool(F.relu(emg_conv(emg)))

        # IMU pathway
        imu = imu.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 9)
        imu = imu.view(imu.size(0), 1, -1)  # Reshape to (batch_size, 1, 9)
        for imu_conv in self.imu_conv_layers:
            imu = self.imu_pool(F.relu(imu_conv(imu)))

        # Flatten outputs
        emg = emg.view(emg.size(0), -1)
        imu = imu.view(imu.size(0), -1)

        # Concatenate EMG and IMU outputs
        concat_out = torch.cat((emg, imu), dim=1)

        # Fully connected layer
        output = self.fc_concat(concat_out)
        output = self.dropout(output)
        # output = self.softmax(output)

        return output

def train_cnn(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        exact_matches = 0
        
        for emg, imu, label in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(emg, imu)

            # Compute loss
            loss = criterion(outputs, label)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()

            # Exact match ratio calculation
            exact_matches += (predicted == labels_idx).sum().item() == label.size(0)

        # Compute average loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        emr = exact_matches / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, EMR: {emr}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

    return train_loss, train_accuracy

def evaluate_cnn(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for emg, imu, label in val_loader:
            outputs = model(emg, imu)
            loss = criterion(outputs, label)
            running_loss += loss.item()

            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()
            y_true.extend(labels_idx.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total

    return val_loss, val_accuracy, y_true, y_pred

def inference_cnn(model, emg_input, imu_input):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(emg_input, imu_input)
        _, predicted = torch.max(output, 1)
    return predicted

class MyNetworkCNN(nn.Module):
    def __init__(self, input_shape_emg, num_classes, hidden_sizes_emg, dropout_rate):
        super(MyNetworkCNN, self).__init__()
        self.emg_input_shape = input_shape_emg
        self.num_classes = num_classes
        self.hidden_sizes_emg = hidden_sizes_emg
        self.dropout_rate = dropout_rate

        # EMG pathway using CNN
        self.emg_conv_layers = nn.ModuleList()
        self.emg_pool = nn.MaxPool1d(kernel_size=2)
        emg_input_channels = 1  # Since EMG input is (batch, 11, 4), treat it as 1-channel signal
        emg_output_size = self.emg_input_shape[0] * self.emg_input_shape[1]
        for emg_hidden_size in self.hidden_sizes_emg:
            self.emg_conv_layers.append(nn.Conv1d(emg_input_channels, emg_hidden_size, kernel_size=3, padding=1))
            emg_input_channels = emg_hidden_size
            emg_output_size = emg_output_size // 2  # Assuming MaxPooling reduces by half each time

        # Fully connected layer
        fc_input_size = emg_output_size * self.hidden_sizes_emg[-1]
        self.fc_concat = nn.Linear(fc_input_size, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emg):
        # EMG pathway
        emg = emg.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 11, 4)
        emg = emg.view(emg.size(0), 1, -1)  # Reshape to (batch_size, 1, 44)
        for emg_conv in self.emg_conv_layers:
            emg = self.emg_pool(F.relu(emg_conv(emg)))

        # Flatten output
        emg = emg.view(emg.size(0), -1)

        # Fully connected layer
        output = self.fc_concat(emg)
        output = self.dropout(output)
        # output = self.softmax(output)

        return output

def train_EMG_cnn(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        exact_matches = 0
        
        for emg, label in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(emg)

            # Compute loss
            loss = criterion(outputs, label)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()

            # Exact match ratio calculation
            exact_matches += (predicted == labels_idx).sum().item() == label.size(0)

        # Compute average loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        emr = exact_matches / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, EMR: {emr}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

    return train_loss, train_accuracy

def evaluate_EMG_cnn(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for emg, label in val_loader:
            outputs = model(emg)
            loss = criterion(outputs, label)
            running_loss += loss.item()

            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()
            y_true.extend(labels_idx.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total

    return val_loss, val_accuracy, y_true, y_pred
    
#%% FFNN network

class Network(nn.Module):
    def __init__(self, **config):
        super(Network, self).__init__(**config)
        print(config)

class MyMultimodalNetwork(nn.Module):
    def __init__(self, input_shape_emg, input_shape_imu, num_classes, hidden_sizes_emg, hidden_sizes_imu, dropout_rate):
        super().__init__()
        self.emg_input_shape = input_shape_emg
        self.imu_input_shape = input_shape_imu
        self.num_classes = num_classes
        self.hidden_sizes_emg = hidden_sizes_emg
        self.hidden_sizes_imu = hidden_sizes_imu
        self.dropout_rate = dropout_rate

        # EMG pathway
        self.emg_flatten = nn.Flatten()
        self.emg_hidden_layers = nn.ModuleList()
        self.emg_dropout_layers = nn.ModuleList()

        if isinstance(self.emg_input_shape, tuple):
            emg_in_features = np.prod(self.emg_input_shape)  # Flatten the input
        else:
            emg_in_features = self.emg_input_shape  # Already a flat number of features
        for emg_hidden_size in self.hidden_sizes_emg:
            self.emg_hidden_layers.append(nn.Linear(emg_in_features, emg_hidden_size))
            self.emg_dropout_layers.append(nn.Dropout(self.dropout_rate))
            emg_in_features = emg_hidden_size

        # IMU pathway
        self.imu_flatten = nn.Flatten()
        self.imu_hidden_layers = nn.ModuleList()
        self.imu_dropout_layers = nn.ModuleList()

        if isinstance(self.imu_input_shape, tuple):
            imu_in_features = np.prod(self.imu_input_shape)  # Flatten the input
        else:
            imu_in_features = self.imu_input_shape  # Already a flat number of features

        for imu_hidden_size in self.hidden_sizes_imu:
            self.imu_hidden_layers.append(nn.Linear(imu_in_features, imu_hidden_size))
            self.imu_dropout_layers.append(nn.Dropout(self.dropout_rate))
            imu_in_features = imu_hidden_size

        # Fully connected layer for concatenation
        fc_input_size = self.hidden_sizes_emg[-1] + self.hidden_sizes_imu[-1]
        self.fc_concat = nn.Linear(fc_input_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emg, imu):
        # EMG pathway
        emg = self.emg_flatten(emg)
        for emg_layer, emg_dropout in zip(self.emg_hidden_layers, self.emg_dropout_layers):
            emg = torch.relu(emg_layer(emg))
            emg = emg_dropout(emg)

        # IMU pathway
        imu = self.imu_flatten(imu)
        for imu_layer, imu_dropout in zip(self.imu_hidden_layers, self.imu_dropout_layers):
            imu = torch.relu(imu_layer(imu))
            imu = imu_dropout(imu)

        # Concatenate EMG and IMU outputs
        concat_out = torch.cat((emg, imu), dim=1)

        # Fully connected layer
        output = self.fc_concat(concat_out)
        output = self.softmax(output)

        return output

def train_binary(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for emg, imu, label in train_loader:
            optimizer.zero_grad()
            outputs = model(emg, imu)
            print(outputs.size())
            print(label.size())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")
    return train_loss, train_accuracy

def train_multiclass(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        exact_matches = 0
        for emg, imu, label in train_loader:
            optimizer.zero_grad()
            outputs = model(emg, imu)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            # label.T for multiclass classification problem
            correct += (predicted == labels_idx).sum().item()
            exact_matches += (predicted == labels_idx).all().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        emr = exact_matches / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, EMR: {emr}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")
    return train_loss, train_accuracy

def test_binary(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for emg, imu, label in test_loader:
            outputs = model(emg, imu)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")
    return test_accuracy

def test_multiclass(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for emg, imu, label in test_loader:
            outputs = model(emg, imu)
            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()
            y_true.extend(labels_idx.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")
    return test_accuracy, y_true, y_pred

def test_multi_and_log(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for emg, imu, label in test_loader:
            outputs = model(emg, imu)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    test_accuracy = correct / total
    wandb.log({"test_accuracy": test_accuracy})

def get_online_tensor_dataset(emg_data, imu_data, labels, config):
    if config["model_type"] == "lstm" and config["input_type"] == "emg+raw_imu":
        emg_data = np.array(emg_data).reshape(1, num_emg_channels*4)
    else:
        emg_data = np.array(emg_data).reshape(1, num_emg_channels, 4)
    if imu_data.shape[0] == 18:
        imu_data = np.array(imu_data).reshape(1, 18)
    elif imu_data.shape[0] == 9:
        imu_data = np.array(imu_data).reshape(1, 9) #5
    label_data = np.array(labels).reshape(1, 5)
    assert len(emg_data) == len(imu_data) == len(label_data)
    emg_tensor = torch.tensor(emg_data, dtype=torch.float32)
    imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
    label_tensor = torch.tensor(label_data, dtype=torch.float32)
    return TensorDataset(emg_tensor, imu_tensor, label_tensor)

#%% EMG network

class NetworkEMG(nn.Module):
    def __init__(self, **config):
        super(Network, self).__init__(**config)
        print(config)

class MyEMGNetwork(nn.Module):
    def __init__(self, input_shape_emg, num_classes, hidden_sizes_emg):
        super().__init__()
        self.emg_input_shape = input_shape_emg
        self.num_classes = num_classes
        self.hidden_sizes_emg = hidden_sizes_emg

        # EMG pathway
        self.emg_flatten = nn.Flatten()
        self.emg_hidden_layers = nn.ModuleList()
        emg_in_features = self.emg_input_shape[0] * self.emg_input_shape[1]
        for emg_hidden_size in self.hidden_sizes_emg:
            self.emg_hidden_layers.append(nn.Linear(emg_in_features, emg_hidden_size))
            emg_in_features = emg_hidden_size

        # Fully connected layer for concatenation
        fc_input_size = self.hidden_sizes_emg[-1]
        self.fc_concat = nn.Linear(fc_input_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emg):
        # EMG pathway
        emg = self.emg_flatten(emg)
        for emg_layer in self.emg_hidden_layers:
            emg = torch.relu(emg_layer(emg))

        # Fully connected layer
        output = self.fc_concat(emg)
        output = self.softmax(output)

        return output
    
def train_EMG(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        exact_matches = 0
        for emg, label in train_loader:
            optimizer.zero_grad()
            outputs = model(emg)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            labels_idx = torch.argmax(label, dim=1)
            correct += (predicted == labels_idx).sum().item()
            exact_matches += (predicted == labels_idx).all().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        emr = exact_matches / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, EMR: {emr}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")
    return train_loss, train_accuracy

def test_EMG(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    exact_matches = 0
    with torch.no_grad():
        for emg, label in test_loader:
            outputs = model(emg)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            labels_idx = torch.argmax(label, dim=1)
            correct += (predicted == labels_idx).sum().item()
            exact_matches += (predicted == labels_idx).all().item()
    test_accuracy = correct / total
    print(f"Test EMR: {exact_matches / len(test_loader)}")
    print(f"Test accuracy: {test_accuracy}")
    return test_accuracy
#%% IMPORT LIBRARIES, SEED, AND GPU INFO
import os
import random
import numpy as np
from scipy import signal
from vqf import offlineVQF
from scipy.spatial.transform import Rotation
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# IMPORT CUSTOM MODULES
from cbpr_master_thesis.preprocessing_and_normalization import notch_filter, bandpass_filter, highpass_filter, lowpass_filter, normalize_EMG_all_channels, convert_to_SI, save_max_emg_values
from cbpr_master_thesis.feature_extraction import create_windows, get_new_feature_vector, extract_EMG_features, extract_quaternions, extract_quaternions_new, extract_angles_from_rot_matrix
from cbpr_master_thesis.model import MyMultimodalNetwork, MyEMGNetwork, train_EMG, test_EMG, get_tensor_dataset, train_multiclass, test_multiclass
from cbpr_master_thesis.data_analysis import plot_confusion_matrix, print_classification_report#, plot_feature_importance

DEFAULT_RANDOM_SEED = 0

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("No GPU available")

# gpu info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)

#%% FUNCTIONS

def read(data_dir, name):
    filename = name + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return x, y

# LOAD AND LOG
'''
It loads the data and the labels from the locally saved numpy files.
If wandb_enabled is True, the function will log the raw data to W&B. 
Otherwise, it will simply return the data.
'''
def load_and_log(name_dataset, name_labels, wandb_enabled=False, no_weight_classification=True, wandb_project=None):
    if wandb_enabled:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name, job_type="load-data") as run:
            # LOAD THE DATASET
            data, label = load(no_weight_classification, name_dataset, name_labels)
            raw_data = wandb.Artifact(
                "raw-data", type="dataset",
                description="Raw EMG and IMU data directly from the sensors",
                metadata={"source": "Delsys Trigno System",
                        "size": [np.array(data).shape]})
            # SAVE THE DATASET AND THE LABELS
            with raw_data.new_file("raw_data.pt", mode="wb") as file:
                torch.save((data, label), file)
            run.log_artifact(raw_data)
        return data, label
    else:
        return load(no_weight_classification, name_dataset, name_labels)

def load(no_weight_classification, name_dataset, name_labels):
    save_path = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/' + name_dataset
    save_path_label = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/' + name_labels
    data = np.load(save_path)
    label = np.load(save_path_label)
    if no_weight_classification:
        for i in range(len(label)):
            if label[i] == 1 or label[i] == 2:
                label[i] = 0
            elif label[i] == 3 or label[i] == 4:
                label[i] = 1
            else:
                print("Error")
    for i in range(len(label)):
        label[i] = label[i] - 1
    return data, label

# PREPROCESS AND LOG
'''
If we don't pass any data, the function will download the latest artifact from the "raw-data" artifact 
and preprocess it.
We don't log the preprocessed data to W&B, but we return it.
'''
def preprocess_and_log(num_classes, data=None, label=None, wandb_project=None):
    if data is None and label is None:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name) as run:
            raw_data_artifact = run.use_artifact('raw-data:latest')
            raw_dataset_dir = raw_data_artifact.download()
            raw_dataset, label = read(raw_dataset_dir, "raw_data")
    else:
        raw_dataset = data
    window_length = 200  # Change this value as needed
    overlap = 100
    return preprocess_and_window(raw_dataset, label, window_length, overlap, num_classes)

def preprocess_and_window(data, label, window_length, overlap, num_classes):
    # Preprocess the EMG data
    data[:,:11,:] = notch_filter(bandpass_filter(highpass_filter(data[:,:11,:], 0.5), 0.5, 100), 50, 30)
    # Convert the IMU data to SI units
    data = convert_to_SI(data)
    # (movements, windows_same_movement, channels, samples) (movements, windows_same_movement)
    windows, label = create_windows(data, window_length, overlap, label)
    one_hot_matrix = np.zeros((label.shape[0], label.shape[1], num_classes))
    label = label.astype(int)
    for movement in range(label.shape[0]):
        for i in range(label.shape[1]):
            index = label[movement, i]
            one_hot_matrix[movement, i, index] = 1.
    # (movements, windows_same_movement, num_classes)
    label = one_hot_matrix
    # (windows, num_classes)
    label = label.reshape(-1, num_classes)
    return windows, label

def train_val_dataset_and_log(wandb_enabled=False, windows=None, labels=None, multiclass_problem=False, wandb_project=None):
    if windows is None and labels is None:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name) as run:
            raw_data_artifact = run.use_artifact('raw-data:latest')
            raw_dataset_dir = raw_data_artifact.download()
            raw_dataset, label = read(raw_dataset_dir, "raw_data")
            window_lenght = 100  # Change this value as needed
            overlap = 80
            windows, labels = preprocess_and_window(raw_dataset, label, window_lenght, overlap, num_classes=4)
            emg_vector, imu_vector = get_new_feature_vector(windows)
            training_dataset, validation_dataset = get_tensor_dataset(emg_vector, imu_vector, labels, multiclass_problem)
    else:
        emg_vector, imu_vector = get_new_feature_vector(windows)
        training_dataset, validation_dataset = get_tensor_dataset(emg_vector, imu_vector, labels, multiclass_problem)
    if wandb_enabled:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name) as run:
            train_val_dataset = wandb.Artifact(
                "train-val-dataset", type="dataset",
                description="Training and validation datasets")
            with train_val_dataset.new_file("train_val_dataset.pt", mode="wb") as file:
                torch.save((training_dataset, validation_dataset), file)
            run.log_artifact(train_val_dataset)
    return training_dataset, validation_dataset

def train_multi(model, train_loader, criterion, optimizer, num_epochs, patience=64):
    model.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
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
            correct += (predicted == labels_idx).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Validation phase
        #val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        # Log metrics using wandb
        wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Train Accuracy": train_accuracy,})
#                   "Validation Loss": val_loss, "Validation Accuracy": val_accuracy

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, ")
#              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping check
        early_stopping(train_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load the best model before returning
    model.load_state_dict(early_stopping.best_model_wts)
    return train_loss, train_accuracy

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for emg, imu, label in val_loader:
            outputs = model(emg, imu)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            labels_idx = torch.argmax(label, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == labels_idx).sum().item()

    val_loss = running_loss / total
    val_accuracy = correct / total
    return val_loss, val_accuracy

def test_multi(model, test_loader):
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
            #y_true.extend(labels_idx.cpu().numpy())
            #y_pred.extend(predicted.cpu().numpy())
    
    test_accuracy = correct / total
    
    # Log test accuracy
    wandb.log({"Test Accuracy": test_accuracy})
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy#, y_true, y_pred

#%% HYPERPARAMETER TUNING

dataset_name='prova_senza5_dataset.npy'
label_name='prova_senza5_labels.npy'
num_classes=6
data, label = load_and_log(dataset_name, label_name, no_weight_classification=False, wandb_enabled=False, wandb_project="tuning_test")
windows, label = preprocess_and_log(num_classes=num_classes, data=data, label=label)
train_data, val_data = train_val_dataset_and_log(windows=windows, labels=label, wandb_enabled=False, wandb_project="tuning_test")

#%%

sweep_config = {
    'method': 'random',
    'metric': {'name': 'Test Accuracy', 'goal': 'maximize'},
    }

# Hyperparameter Tuning
parameters_dict = {
    'optimizer': {
        'values': ['sgd', 'adam']},
    'hidden_sizes_emg': {
        'values': [[512, 2048, 2048, 2048, 1024, 1024, 1024, 512], [512, 2048, 2048, 1024, 1024, 1024, 512], [512, 2048, 2048, 1024, 1024, 512], [512, 1024, 1024, 1024, 1024, 1024, 512], [512, 2048, 2048, 2048, 2048, 2048, 512], [512, 512, 512, 512, 512, 512, 512],]},
    'hidden_sizes_imu': {
        'values': [[512, 2048, 2048, 2048, 1024, 1024, 1024, 512], [512, 2048, 2048, 1024, 1024, 1024, 512], [512, 2048, 2048, 1024, 1024, 512], [512, 1024, 1024, 1024, 1024, 1024, 512], [512, 2048, 2048, 2048, 2048, 2048, 512], [512, 512, 512, 512, 512, 512, 512],]},
    'dropout_rate': {
        'values': [0.1, 0.2, 0.3]},
    'learning_rate': {
        'distribution': 'uniform', 'min': 0.001, 'max': 0.05}, # maybe even smaller
    'batch_size': {
        'values': [64, 128, 256]},
    'weight_decay': {  # L2 regularization
        'distribution': 'log_uniform', 'min': 1e-5, 'max': 1e-3},
    'epochs': {
        'value': 256},
    'input_shape_emg': {
        'value': (11, 4)},
    'input_shape_imu': {
        'value': 9},
    'num_classes': {
        'value': 6}
}

sweep_config['parameters'] = parameters_dict

import pprint
pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="tuning_test")

# Training function
def train_hyperparameter_tuning(config=None):
    global train_data, val_data
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        # DataLoader
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        validation_loader = DataLoader(val_data, batch_size=config.batch_size)
        # Model
        model = build_model(config)
        # Loss Function
        criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multiclass classification
        # Optimizer
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        # Call train_multi with early stopping
        #train_loss, train_accuracy = train_multiclass(model, train_loader, criterion, optimizer, config.epochs)#, scheduler)
        train_loss, train_accuracy = train_multi(model, train_loader, criterion, optimizer, config.epochs, patience=64)
        wandb.log({"Final Training Loss": train_loss, "Final Training Accuracy": train_accuracy})
        test_accuracy = test_multi(model, validation_loader)
        wandb.log({"Test Accuracy": test_accuracy})

def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def build_model(config):
    model = MyMultimodalNetwork(input_shape_emg=config.input_shape_emg, 
                                input_shape_imu=config.input_shape_imu, 
                                num_classes=config.num_classes, 
                                hidden_sizes_emg=config.hidden_sizes_emg, 
                                hidden_sizes_imu=config.hidden_sizes_imu, 
                                dropout_rate=config.dropout_rate)
    return model

# Early Stopping Implementation
class EarlyStopping:
    def __init__(self, patience=64, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0

# Run the hyperparameter tuning
wandb.agent(sweep_id, train_hyperparameter_tuning, count=50)

#%% HYPERPARAMETER TUNING WITH ONLY EMG

data, label = load_and_log(wandb_enabled=False)
windows, label = preprocess_and_log(data, label)
windows_EMG = []
for w in windows:
    windows_EMG.append(w[:11,:])
    
emg_vector = normalize_EMG_all_channels(extract_EMG_features(windows))
emg_data = np.array(emg_vector)
label_data = np.array(label)
assert len(emg_data) == len(label_data)
emg_tensor = torch.tensor(emg_data, dtype=torch.float32)
label_tensor = torch.tensor(label_data, dtype=torch.long)
# Split the data into training and validation sets
emg_train, emg_val, labels_train, labels_val = train_test_split(emg_tensor, label_tensor, test_size=0.2, random_state=42)
train_data = TensorDataset(emg_train, labels_train) 
val_data = TensorDataset(emg_val, labels_val)


sweep_config = {
    'method': 'random',
    'metric': {'name': 'Test Accuracy', 
               'goal': 'maximize'},
    }

parameters_dict = {
    'optimizer': {
        'value': 'sgd'},
    'hidden_sizes_emg': {
        'values': [[44, 64, 64, 64, 32, 8], [44, 32, 16, 16, 8, 4], [44, 64, 128, 128, 64, 4]]},
    'hidden_sizes_imu': {
        'values': [[5, 8, 16, 16, 16, 8, 4], [5, 16, 32, 32, 8, 4], [5, 32, 64, 64, 32, 4]]}
    }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'epochs': {
        'value': 16},
    'input_shape_emg': {
        'value': (11,4)},
    'input_shape_imu': {
        'value': 5},
    'num_classes': {
        'value': 2}
    })

parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.09
      },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })

import pprint
pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="SL_pipeline_EMG")

def train_hyperparameter_tuning(config=None):
    global train_data, val_data
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        # train loader
        train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
        # model
        model = build_model_EMG(config)
        # optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
        train_accuracy, train_loss = train_multi_EMG(model, train_loader, criterion, optimizer, config.epochs)
        wandb.log({"Loss": {train_loss}, "Accuracy": {train_accuracy}})
        validation_loader = DataLoader(val_data, batch_size=128)
        test_accuracy = test_multi_EMG(model, validation_loader)
        wandb.log({"Test Accuracy": test_accuracy})


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    return optimizer

def build_model_EMG(config):
    model = MyEMGNetwork(input_shape_emg=config['input_shape_emg'], num_classes=config['num_classes'], hidden_sizes_emg=config['hidden_sizes_emg'])
    return model

wandb.agent(sweep_id, train_hyperparameter_tuning, count=50)

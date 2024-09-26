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
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# IMPORT CUSTOM MODULES
from cbpr_master_thesis.preprocessing_and_normalization import notch_filter, bandpass_filter, highpass_filter, lowpass_filter, normalize_EMG_all_channels, convert_to_SI, save_max_emg_values, normalize_raw_imu
from cbpr_master_thesis.feature_extraction import create_windows, get_new_feature_vector, extract_EMG_features, extract_quaternions, extract_quaternions_new, extract_angles_from_rot_matrix
from cbpr_master_thesis.model import MyMultimodalNetwork, MyEMGNetwork, train_EMG, test_EMG, get_tensor_dataset, train_multiclass, test_multiclass, test_and_storing, test_EMG_and_storing
from cbpr_master_thesis.data_analysis import undersample_majority_class_first_n, extract_and_balance, data_analysis, plot_accuracy_boxplots, plot_performance_lineplot, plot_f1_scores_barchart, comparison_analysis, plot_heatmap, bar_chart_with_error_bars

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

'''
It loads the data and the labels from the locally saved numpy files.
If wandb_enabled is True, the function will log the raw data to W&B. 
Otherwise, it will simply return the data.
'''
def load_and_log(name_dataset, name_labels, wandb_enabled=False, no_weight_classification=False, wandb_project=None):
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
    window_length = 200  # Samples
    overlap = 100
    return preprocess_and_window(raw_dataset, label, window_length, overlap, num_classes)

def preprocess_and_window(data, label, window_length, overlap, num_classes):
    global num_emg_channels
    # Preprocess the EMG data
    data[:,:num_emg_channels,:] = notch_filter(bandpass_filter(highpass_filter(data[:,:num_emg_channels,:], 0.5), 0.5, 100), 50, 30)
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

'''
If wandb_enabled is True, the function will log the model to W&B.
In any case, it will return the model.
'''
def build_model_and_log(config, wandb_enabled=False, wandb_project=None):
    if wandb_enabled:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name, job_type="initialize") as run:
            model = MyMultimodalNetwork(input_shape_emg=config['input_shape_emg'], input_shape_imu=config['input_shape_imu'], num_classes=config['num_classes'], hidden_sizes_emg=config['hidden_sizes_emg'], hidden_sizes_imu=config['hidden_sizes_imu'], dropout_rate=0.1)
            model_artifact = wandb.Artifact(
                "network", type="model",
                description="Simple neural network")
            with model_artifact.new_file("init_model.pth", mode="wb") as file:
                torch.save(model.state_dict(), file)
            run.log_artifact(model_artifact)
    else:
        model = MyMultimodalNetwork(input_shape_emg=config['input_shape_emg'], input_shape_imu=config['input_shape_imu'], num_classes=config['num_classes'], hidden_sizes_emg=config['hidden_sizes_emg'], hidden_sizes_imu=config['hidden_sizes_imu'], dropout_rate=0.1)
    return model

'''
If we don't pass any data, the function will download the latest artifact from the "raw-data" artifact, preprocess it, 
and create the training and validation datasets.
If wandb_enabled is True, we log the training and validation datasets to W&B.
'''
def train_val_dataset_and_log(wandb_enabled=False, windows=None, labels=None, wandb_project=None):
    if windows is None and labels is None:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name) as run:
            raw_data_artifact = run.use_artifact('raw-data:latest')
            raw_dataset_dir = raw_data_artifact.download()
            raw_dataset, label = read(raw_dataset_dir, "raw_data")
            window_lenght = 200  # Change this value as needed
            overlap = 100
            windows, labels = preprocess_and_window(raw_dataset, label, window_lenght, overlap, num_classes=4)
            emg_vector, imu_vector = get_new_feature_vector(windows)
            training_dataset, validation_dataset = get_tensor_dataset(emg_vector, imu_vector, labels)
    else:
        emg_vector, imu_vector = get_new_feature_vector(windows, labels)
        training_dataset, validation_dataset = get_tensor_dataset(emg_vector, imu_vector, labels)
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

def train_val_dataset_raw_imu(windows, labels):
    global num_emg_channels
    windows = windows.reshape(windows.shape[0]*windows.shape[1], windows.shape[2], windows.shape[3])
    imu_vector = normalize_raw_imu(windows[:,num_emg_channels:,:])
    # Compute the mean across windows and samples for each channel
    imu_vector = np.mean(imu_vector, axis=2).squeeze()
    emg_vector = np.array(normalize_EMG_all_channels(extract_EMG_features(windows)))
    training_dataset, validation_dataset = get_tensor_dataset(emg_vector, imu_vector, labels)
    return training_dataset, validation_dataset

'''
If we don't pass any data, the function will download the latest artifact from the "train-val-dataset" artifact, and the latest artifact from the "network" artifact.
If wandb_enabled is True, we log the trained model to W&B.
In any case, we return the trained model.
'''
def train_and_log(train_config, model_config, training_dataset=None, model=None, wandb_enabled=False, criterion=None, wandb_project=None):
    if training_dataset is None:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name, job_type="train", config=train_config) as run:
            train_config = wandb.config
            datasets = run.use_artifact('train-val-dataset:latest')
            datasets_dir = datasets.download()
            training_dataset, _ = read(datasets_dir, "train_val_dataset")
            if model is None:
                model_artifact = run.use_artifact("network:latest")
                model_dir = model_artifact.download()
                model_path = os.path.join(model_dir, "init_model.pth")
                model_config = model_artifact.metadata
                train_config.update(model_config)
                model = MyMultimodalNetwork(input_shape_emg=model_config['input_shape_emg'], input_shape_imu=model_config['input_shape_imu'], num_classes=model_config['num_classes'], hidden_sizes_emg=model_config['hidden_sizes_emg'], hidden_sizes_imu=model_config['hidden_sizes_imu'], dropout_rate=0.5)
                model.load_state_dict(torch.load(model_path))
                model = model.to(device)
    if train_config["criterion"] == 'bce_with_logits':
        criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    else:
        criterion = nn.CrossEntropyLoss()
    if train_config["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=0.9)
    train_loader = DataLoader(training_dataset, batch_size=train_config['batch_size'])
    train_multiclass(model, train_loader, criterion, optimizer, train_config['epochs'])
    if wandb_enabled:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name) as run:
            model_artifact = wandb.Artifact(
                "trained-network", type="model",
                description="Trained NN model",
                #metadata=dict(model_config)
                )
            with model_artifact.new_file("trained_model.pth", mode="wb") as file:
                torch.save(model.state_dict(), file)
            run.log_artifact(model_artifact)
    return model

'''
If we don't pass any data, the function will download the latest artifact from the "train-val-dataset" artifact, and the latest artifact from the "trained-network" artifact.
'''
def evaluate_and_log(config, validation_dataset=None, model=None, wandb_project=None):
    if validation_dataset is None:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name, job_type="report", config=config) as run:
            datasets = run.use_artifact('train-val-dataset:latest')
            datasets_dir = datasets.download()
            _, validation_dataset = read(datasets_dir, "train_val_dataset")
            if model is None:
                model_artifact = run.use_artifact("trained-network:latest")
                model_dir = model_artifact.download()
                model_path = os.path.join(model_dir, "trained_model.pth")
                model_config = model_artifact.metadata
                model = MyMultimodalNetwork(input_shape_emg=model_config['input_shape_emg'], input_shape_imu=model_config['input_shape_imu'], num_classes=model_config['num_classes'], hidden_sizes_emg=model_config['hidden_sizes_emg'], hidden_sizes_imu=model_config['hidden_sizes_imu'], dropout_rate=0.5)
                model.load_state_dict(torch.load(model_path))
                model = model.to(device)
    validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'])
    return test_multiclass(model, validation_loader)

#%% PIPELINES

def pipeline(dataset_name, label_name, num_classes):
    data, label = load_and_log(dataset_name, label_name, no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
    # (movements, windows_same_movement, channels, samples) (windows, num_classes) label is one hot encoded
    windows, label = preprocess_and_log(num_classes=num_classes, data=data, label=label)
    config = {
        'num_classes': num_classes,
        'hidden_sizes_emg': [256, 256, 256],#[512, 2048, 2048, 2048, 2048, 2048, 512],
        'hidden_sizes_imu': [256, 256, 256],#[512, 2048, 2048, 2048, 2048, 2048, 512],
        'input_shape_emg': (num_emg_channels, 4),
        'input_shape_imu': 9, #9
        'dropout_rate': 0.1
    }
    model = build_model_and_log(config, wandb_enabled=False, wandb_project="SL_multiclass")
    train_data, val_data = train_val_dataset_and_log(windows=windows, labels=label, wandb_enabled=False, wandb_project="SL_multiclass")
    train_config = {"batch_size": 128,
                    "epochs": int(32), #256 -> 99.5%
                    "criterion": "",
                    "optimizer": "adam",
                    "learning_rate": 0.001} #0.05
    # or weighted BCE
    trained_model = train_and_log(train_config, config, training_dataset=train_data, model=model, wandb_enabled=False, wandb_project="SL_multiclass")
    test_accuracy, y_true, y_pred = evaluate_and_log(train_config, val_data, trained_model, wandb_project="SL_multiclass")
    #plot_confusion_matrix(y_true, y_pred)
    #print_classification_report(y_true, y_pred)
    #plot_feature_importance(trained_model, train_data[:][0], train_data[:][1])
    return trained_model

def pipeline_raw_IMU(dataset_name, label_name, num_classes):
    data, label = load_and_log(dataset_name, label_name, no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
    # (movements, windows_same_movement, channels, samples) (windows, num_classes) label is one hot encoded
    windows, label = preprocess_and_log(num_classes=num_classes, data=data, label=label)
    config = {
        'num_classes': num_classes,
        'hidden_sizes_emg': [256, 256, 256],#[512, 2048, 2048, 2048, 2048, 2048, 512],
        'hidden_sizes_imu': [256, 256, 256],#[512, 2048, 2048, 2048, 2048, 2048, 512],
        'input_shape_emg': (num_emg_channels, 4),
        'input_shape_imu': 18, #9
        'dropout_rate': 0.1
    }
    model = build_model_and_log(config, wandb_enabled=False, wandb_project="SL_multiclass")
    train_data, val_data = train_val_dataset_raw_imu(windows=windows, labels=label)
    train_config = {"batch_size": 32,
                    "epochs": 32,
                    "criterion": "",
                    "optimizer": "adam",
                    "learning_rate": 0.001} #0.05
    # Create data loaders
    model = train_and_log(train_config=train_config, model_config=config, training_dataset=train_data, model=model, wandb_enabled=False, wandb_project=None)
    # Evaluate the model on validation set
    val_accuracy, y_true, y_pred = evaluate_and_log(train_config, validation_dataset=val_data, model=model)
    print(f"Validation Accuracy: {val_accuracy}")
    #plot_confusion_matrix(y_true, y_pred)
    return model

def pipeline_EMG(dataset_name, label_name, num_classes):
    data, label = load_and_log(dataset_name, label_name, no_weight_classification=False, wandb_enabled=False)
    windows, label = preprocess_and_log(num_classes=num_classes, data=data, label=label)
    windows = windows.reshape(windows.shape[0]*windows.shape[1], windows.shape[2], windows.shape[3])
    windows_EMG = []
    for w in windows:
        windows_EMG.append(w[:num_emg_channels,:])
    emg_vector = normalize_EMG_all_channels(extract_EMG_features(windows_EMG))
    emg_data = np.array(emg_vector)
    label_data = np.array(label)
    assert len(emg_data) == len(label_data)
    emg_tensor = torch.tensor(emg_data, dtype=torch.float32)
    label_tensor = torch.tensor(label_data, dtype=torch.float32)
    # Split the data into training and validation sets
    emg_train, emg_val, labels_train, labels_val = train_test_split(emg_tensor, label_tensor, test_size=0.2, random_state=42)
    train_data = TensorDataset(emg_train, labels_train) 
    val_data = TensorDataset(emg_val, labels_val)
    config_EMG = {
        'num_classes': num_classes,
        'hidden_sizes_emg': [512, 256, 256, 256],
        'input_shape_emg': (num_emg_channels, 4),
        'dropout_rate': 0.1
    }
    model = MyEMGNetwork(input_shape_emg=config_EMG['input_shape_emg'], num_classes=config_EMG['num_classes'], hidden_sizes_emg=config_EMG['hidden_sizes_emg'])
    train_config_EMG = {
        "batch_size": 32,
        "epochs": 32,
        "criterion": "",
        "optimizer": "sgd",
        "learning_rate": 0.1}
    train_loader = DataLoader(train_data, train_config_EMG['batch_size'], shuffle=True)
    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=train_config_EMG['learning_rate'], momentum=0.9)
    train_accuracy, train_loss = train_EMG(model, train_loader, criterion, optimizer, train_config_EMG['epochs'], multiclass=True)
    validation_loader = DataLoader(val_data, batch_size=32)
    test_accuracy, y_true, y_pred = test_EMG(model, validation_loader, multiclass=True)
    #plot_confusion_matrix(y_true, y_pred)
    #print_classification_report(y_true, y_pred)
    return model

trained_model = pipeline(dataset_name='prova_senza5_dataset.npy', label_name='prova_senza5_labels.npy', num_classes=6)
'''
model_path = 'model_parameters_ffnn.pth'
torch.save(trained_model.state_dict(), model_path)
trained_model_EMG = pipeline_EMG(dataset_name='prova_senza5_dataset.npy', label_name='prova_senza5_labels.npy', num_classes=6)
model_path = 'model_parameters_ffnn_emg.pth'
torch.save(trained_model_EMG.state_dict(), model_path)
trained_model_raw_imu = pipeline_raw_IMU(dataset_name='prova_senza5_dataset.npy', label_name='prova_senza5_labels.npy', num_classes=6)
model_path = 'model_parameters_ffnn_raw_imu.pth'
torch.save(trained_model_raw_imu.state_dict(), model_path)
'''

#%% Pipeline from online recordings

num_emg_channels = 9
global_epochs = 64
base_folder = "C:/Users/claud/Desktop/CBPR_Recordings/"

def pipeline_from_online(emg, imu, label, num_classes, model_path=None, save=False, model_path_save=None, participant_folder=None):
    #emg, imu, label, pred = load_data_from_online(dataset_name)
    training_dataset, validation_dataset = get_tensor_dataset(emg, imu.squeeze(), label.squeeze())
    config = {
        'num_classes': num_classes,
        'hidden_sizes_emg': [256, 256, 256],
        'hidden_sizes_imu': [256, 256, 256],
        #'hidden_sizes_emg': [512, 1024, 1024, 1024, 512],
        #'hidden_sizes_imu': [512, 1024, 1024, 1024, 512],
        'input_shape_emg': (num_emg_channels, 4),
        'input_shape_imu': 9, #9
        'dropout_rate': 0.1
    }
    model = build_model_and_log(config, wandb_enabled=False, wandb_project="SL_multiclass")
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    #batch_size ???
    train_config = {"batch_size": 32,
                    "epochs": global_epochs, #256 -> 99.5%
                    "criterion": "",
                    "optimizer": "adam",
                    "learning_rate": 0.001} #0.05
    # or weighted BCE
    trained_model = train_and_log(train_config, config, training_dataset=training_dataset, model=model, wandb_enabled=False, wandb_project="SL_multiclass")
    #test_accuracy, y_true, y_pred, all_predictions = evaluate_and_log(train_config, validation_dataset, trained_model, wandb_project="SL_multiclass")
    validation_loader = DataLoader(validation_dataset, batch_size=train_config['batch_size'])
    test_accuracy, y_true, y_pred, emg_data, imu_data, label_data, predictions = test_and_storing(model, validation_loader)
    #plot_confusion_matrix(y_true, y_pred)
    if save:
        model_path = model_path_save
        torch.save(trained_model, model_path)
    directory = base_folder + participant_folder
    file_name = "ffnn_angles_dataset.npz"
    file_path = os.path.join(directory, file_name) if directory else file_name
    data_dict = {
        'emg': np.array(emg_data),
        'imu': np.array(imu_data),
        'label': np.array(label_data),
        'prediction': np.array(predictions)
    }
    # WAIT FOR SAVE THE DATA
    np.savez(file_path, **data_dict)
    return trained_model

def pipeline_raw_IMU_from_online(emg, imu, label, num_classes, model_path=None, save=False, model_path_save=None, participant_folder=None):
    #emg, imu, label, pred = load_data_from_online(dataset_name)
    training_dataset, validation_dataset = get_tensor_dataset(emg, imu.squeeze(), label.squeeze())
    config = {
        'num_classes': num_classes,
        'hidden_sizes_emg': [256, 256, 256],
        'hidden_sizes_imu': [256, 256, 256],
        #'hidden_sizes_emg': [2048, 2048, 2048, 2048, 2048],
        #'hidden_sizes_imu': [2048, 2048, 2048, 2048, 2048],
        'input_shape_emg': (num_emg_channels, 4),
        'input_shape_imu': 18, #9
        'dropout_rate': 0.1
    }
    model = build_model_and_log(config, wandb_enabled=False, wandb_project="SL_multiclass")
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    training_dataset, validation_dataset = get_tensor_dataset(emg, imu, label)
    train_config = {"batch_size": 32,
                    "epochs": global_epochs,
                    "criterion": "",
                    "optimizer": "adam",
                    "learning_rate": 0.01} #0.05
    # Create data loaders
    trained_model = train_and_log(train_config=train_config, model_config=config, training_dataset=training_dataset, model=model, wandb_enabled=False, wandb_project=None)
    # Evaluate the model on validation set
    validation_loader = DataLoader(validation_dataset, batch_size=train_config['batch_size'])
    test_accuracy, y_true, y_pred, emg_data, imu_data, label_data, predictions = test_and_storing(model, validation_loader)
    #plot_confusion_matrix(y_true, y_pred)
    if save:
        model_path = model_path_save
        torch.save(trained_model, model_path)
    directory = base_folder + participant_folder
    file_name = "ffnn_raw_imu_dataset.npz"
    file_path = os.path.join(directory, file_name) if directory else file_name
    data_dict = {
        'emg': np.array(emg_data),
        'imu': np.array(imu_data),
        'label': np.array(label_data),
        'prediction': np.array(predictions)
    }
    # WAIT FOR SAVE THE DATA
    np.savez(file_path, **data_dict)
    return model

def pipeline_EMG_from_online(emg, label, num_classes, model_path=None, save=False, model_path_save=None, participant_folder=None):
    #emg, label, pred = load_data_from_online(dataset_name)
    emg_data = np.array(emg)
    label_data = np.array(label)
    assert len(emg_data) == len(label_data)
    emg_tensor = torch.tensor(emg_data, dtype=torch.float32)
    label_tensor = torch.tensor(label_data, dtype=torch.float32)
    # Split the data into training and validation sets
    emg_train, emg_val, labels_train, labels_val = train_test_split(emg_tensor, label_tensor, test_size=0.2, random_state=42)
    train_data = TensorDataset(emg_train, labels_train) 
    val_data = TensorDataset(emg_val, labels_val)
    config_EMG = {
        'num_classes': num_classes,
        'hidden_sizes_emg': [256, 512, 256, 256],
        #'hidden_sizes_emg': [512, 1024, 1024, 1024, 512],
        'input_shape_emg': (num_emg_channels, 4),
        'dropout_rate': 0.1
    }
    model = MyEMGNetwork(input_shape_emg=config_EMG['input_shape_emg'], num_classes=config_EMG['num_classes'], hidden_sizes_emg=config_EMG['hidden_sizes_emg'])
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    train_config_EMG = {
        "batch_size": 32,
        "epochs": global_epochs,
        "criterion": "",
        "optimizer": "sgd",
        "learning_rate": 0.1}
    train_loader = DataLoader(train_data, train_config_EMG['batch_size'], shuffle=True)
    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=train_config_EMG['learning_rate'], momentum=0.9)
    train_accuracy, train_loss = train_EMG(model, train_loader, criterion, optimizer, train_config_EMG['epochs'], multiclass=True)
    validation_loader = DataLoader(val_data, batch_size=128)
    test_accuracy, y_true, y_pred, emg_data, label_data, predictions = test_EMG_and_storing(model, validation_loader, multiclass=True)
    #plot_confusion_matrix(y_true, y_pred)
    #print_classification_report(y_true, y_pred)
    if save:
        model_path = model_path_save
        torch.save(model, model_path)
    directory = base_folder + participant_folder
    file_name = "ffnn_emg_dataset.npz"
    file_path = os.path.join(directory, file_name) if directory else file_name
    data_dict = {
        'emg': np.array(emg_data),
        'imu': np.zeros((len(emg_data), 9)),
        'label': np.array(label_data),
        'prediction': np.array(predictions)
    }
    # WAIT FOR SAVE THE DATA
    np.savez(file_path, **data_dict)
    return model

def pipeline_inference_and_storing(emg, label, model_path, imu=None, save=False):
    multimodal = False
    emg_data = np.array(emg)
    label_data = np.array(label)
    emg_tensor = torch.tensor(emg_data.squeeze(), dtype=torch.float32)
    label_tensor = torch.tensor(label_data.squeeze(), dtype=torch.float32)
    if imu is not None:
        imu_data = np.array(imu)
        imu_tensor = torch.tensor(imu_data.squeeze(), dtype=torch.float32)
        tensor_dataset = TensorDataset(emg_tensor, imu_tensor, label_tensor)
        multimodal = True
    else:
        tensor_dataset = TensorDataset(emg_tensor, label_tensor)
    loader = DataLoader(tensor_dataset, batch_size=1)
    model = torch.load(model_path)
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            if multimodal:
                emg, imu, label = batch
                output = model(emg, imu)
            else:
                emg, label = batch
                output = model(emg)
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            labels_idx = torch.argmax(label, dim=1)
    if save:
        model_path = 'current_ffnn_angles.pth'
        torch.save(model, model_path)
    return all_predictions

#%% Code for testing the pipelines on saved data

base_path = "//wsl.localhost/Ubuntu/home/dema/CBPR_Recording_Folders/"
recording_numbers = ["0", "1", "2", "3", "4", "5"]
recording_days = ["day2_1", "day3_1"]

data = {
    'emg_angles': [], 'imu_angles': [], 'label_angles': [],
    'emg_raw_imu': [], 'imu_raw_imu': [], 'label_raw_imu': [],
    'emg_emg': [], 'imu_emg': [], 'label_emg': []
}

#folder_name = "day3_1"
for recording_day in recording_days:
    for recording_number in recording_numbers:
        folder_name = recording_day
        file_paths = {
            'angles': os.path.join(base_path, folder_name, f"DATASET_ANGLE_ESTIMATION_{recording_number}", "angles_estimation_dataset.npz"),
            'raw_imu': os.path.join(base_path, folder_name, f"DATASET_RAW_IMU_{recording_number}", "raw_imu_dataset.npz"),
            'emg': os.path.join(base_path, folder_name, f"DATASET_EMG_{recording_number}", "emg_dataset.npz")
        }

        for file_type, file_path in file_paths.items():
            loaded_data = np.load(file_path, allow_pickle=True)
            
            for key in ['emg', 'imu', 'label']:
                data_key = f"{key}_{file_type}"
                if data_key in data:
                    data[data_key].append(loaded_data[key])
            
            print(f"Loaded {file_type} data:")
            for key in ['emg', 'imu', 'label']:
                print(f"  {key}_{file_type}: {loaded_data[key].shape}")

# Concatenate arrays for each key
for key in data.keys():
    if data[key]:  # Check if the list is not empty
        data[key] = np.concatenate(data[key], axis=0)

# Print the shapes of the concatenated arrays
for key, value in data.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: No data")

# delete the first and the 7th emg channel in each emg dataset, because going from 11 to 9 EMG sensors
data['emg_angles'] = np.delete(data['emg_angles'], [0, 6], axis=1)
data['emg_raw_imu'] = np.delete(data['emg_raw_imu'], [0, 6], axis=1)
data['emg_emg'] = np.delete(data['emg_emg'], [0, 6], axis=1)

# Print the shapes of the concatenated arrays
for key, value in data.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: No data")

BASE_MODEL_PATH = "C:/Users/claud/Documents/GitHub/CBPR_Master_Thesis"
data['emg_angles'], data['imu_angles'], data['label_angles'] = undersample_majority_class_first_n(emg_data=data['emg_angles'], imu_data=data['imu_angles'], labels=data['label_angles'])
model_angles_from_online_ffnn = pipeline_from_online(data['emg_angles'], data['imu_angles'], data['label_angles'], num_classes=5, model_path=None, save=True, model_path_save = "models/model_ffnn9.pth") #BASE_MODEL_PATH+"/model_ffnn.pth"

data['emg_raw_imu'], data['imu_raw_imu'], data['label_raw_imu'] = undersample_majority_class_first_n(emg_data=data['emg_raw_imu'], imu_data=data['imu_raw_imu'], labels=data['label_raw_imu'])
model_raw_imu_from_online_ffnn = pipeline_raw_IMU_from_online(data['emg_raw_imu'], data['imu_raw_imu'], data['label_raw_imu'], num_classes=5, model_path=None, save=True, model_path_save = "models/model_ffnn_raw_imu9.pth")

data['emg_emg'], _, data['label_emg'] = undersample_majority_class_first_n(emg_data=data['emg_emg'], labels=data['label_emg'])
model_emg_from_online_ffnn = pipeline_EMG_from_online(data['emg_emg'], data['label_emg'], num_classes=5, model_path=None, save=True, model_path_save = "models/model_ffnn_emg9.pth")

#%% SAVE/LOAD THE MODEL

model_path_ffnn = '//wsl.localhost/Ubuntu/home/dema/ros2_ws/models/model_ffnn.pth'
model_path_ffnn_raw_imu = '//wsl.localhost/Ubuntu/home/dema/ros2_ws/models/model_ffnn_raw_imu.pth'
model_path_ffnn_emg = '//wsl.localhost/Ubuntu/home/dema/ros2_ws/models/model_ffnn_emg.pth'
torch.save(model_angles_from_online_ffnn.state_dict(), model_path_ffnn)
torch.save(model_raw_imu_from_online_ffnn.state_dict(), model_path_ffnn_raw_imu)
torch.save(model_emg_from_online_ffnn.state_dict(), model_path_ffnn_emg)
#trained_model.load_state_dict(torch.load(model_path))

#%% COUNT PARAMETERS

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(count_parameters(model)*4/1024)


#%% FROM HERE TESTS USEFUL FOR MANAGING AND PROCESSING DATA

folder_name = "day3_1"
recording_number = "5"
file_path_angles = "//wsl.localhost/Ubuntu/home/dema/CBPR_Recording_Folders/"+folder_name+"/DATASET_ANGLE_ESTIMATION_"+recording_number+"/angles_estimation_dataset.npz"
file_path_raw_imu = "//wsl.localhost/Ubuntu/home/dema/CBPR_Recording_Folders/"+folder_name+"/DATASET_RAW_IMU_"+recording_number+"/raw_imu_dataset.npz"
file_path_emg = "//wsl.localhost/Ubuntu/home/dema/CBPR_Recording_Folders/"+folder_name+"/DATASET_EMG_"+recording_number+"/emg_dataset.npz"
data = np.load(file_path_angles, allow_pickle=True)
emg_angles, imu_angles, label_angles, pred_ffnn_angles, pred2, pred3 = (data['emg'], data['imu'], data['label'], data['prediction_0'], data['prediction_1'], data['prediction_2'])
pred_ffnn_angles = pred_ffnn_angles.squeeze()
data = np.load(file_path_raw_imu, allow_pickle=True)
emg_raw_imu, imu_raw_imu, label_raw_imu, pred_ffnn_raw_imu, pred2, pred3 = (data['emg'], data['imu'], data['label'], data['prediction_0'], data['prediction_1'], data['prediction_2'])
pred_ffnn_raw_imu = pred_ffnn_raw_imu.squeeze()
data = np.load(file_path_emg, allow_pickle=True)
emg_emg, imu_emg, label_emg, pred_ffnn_emg, pred2, pred3 = (data['emg'], data['imu'], data['label'], data['prediction_0'], data['prediction_1'], data['prediction_2'])
pred_ffnn_emg = pred_ffnn_emg.squeeze()
emg_angles, imu_angles, label_angles = undersample_majority_class_first_n(emg_data=emg_angles, imu_data=imu_angles, labels=label_angles)
predictions1 = pipeline_inference_and_storing(emg_angles, label_angles, 'current_ffnn_angles.pth', imu_angles, save=False)

emg_raw_imu, imu_raw_imu, label_raw_imu = undersample_majority_class_first_n(emg_data=emg_raw_imu, imu_data=imu_raw_imu, labels=label_raw_imu)
predictions2 = pipeline_inference_and_storing(emg_raw_imu, label_raw_imu, 'current_ffnn_raw_imu.pth', imu_raw_imu, save=False)

emg_emg, _, label_emg = undersample_majority_class_first_n(emg_data=emg_emg, labels=label_emg)
predictions3 = pipeline_inference_and_storing(emg_emg, label_emg, 'current_ffnn_emg.pth', save=False)
# dictionary with emg, imu, label and predictions
current_data_angles = {'emg': emg_angles, 'imu': imu_angles, 'label': label_angles, 'predictions': predictions1}
current_data_raw_imu = {'emg': emg_raw_imu, 'imu': imu_raw_imu, 'label': label_raw_imu, 'predictions': predictions2}
current_data_emg = {'emg': emg_emg, 'label': label_emg, 'predictions': predictions3}
np.savez("temp/current_data_angles.npz", **current_data_angles)
np.savez("temp/current_data_raw_imu.npz", **current_data_raw_imu)
np.savez("temp/current_data_emg.npz", **current_data_emg)

#%% FIX DATA: EXTRACT AND BALANCE, SPLIT, TRAIN, TEST AND STORE

number_rec_per_input_type = 2
participant_folders = ["1_30_08", "2_30_08", "3_30_08", "1_31_08", "2_31_08", "3_31_08", "1_02_09", "2_02_09", "1_04_09", "2_04_09", "3_04_09", "1_05_09", "2_05_09", "3_05_09", "1_06_09"]
participant_folders_with_raw_dataset = ["2_02_09", "1_04_09", "2_04_09", "3_04_09", "1_05_09", "2_05_09", "3_05_09", "1_06_09"]

for participant_folder in participant_folders:
    data = {'emg_angles': [], 'imu_angles': [], 'label_angles': [], 
            'emg_raw_imu': [], 'imu_raw_imu': [], 'label_raw_imu': [], 
            'emg_emg': [], 'imu_emg': [], 'label_emg': []}
    data = extract_and_balance(participant_folder, number_rec_per_input_type)
    if data['emg_angles'].shape[1] == 11:
        data['emg_angles'] = np.delete(data['emg_angles'], [0, 6], axis=1)
        data['emg_raw_imu'] = np.delete(data['emg_raw_imu'], [0, 6], axis=1)
        data['emg_emg'] = np.delete(data['emg_emg'], [0, 6], axis=1)
    angles_model = pipeline_from_online(data['emg_angles'], data['imu_angles'], data['label_angles'], num_classes=5, model_path=None, save=False, model_path_save = "models/model_ffnn_angles.pth", participant_folder=participant_folder)
    raw_imu_model = pipeline_raw_IMU_from_online(data['emg_raw_imu'], data['imu_raw_imu'], data['label_raw_imu'], num_classes=5, model_path=None, save=False, model_path_save = "models/model_ffnn_raw_imu.pth", participant_folder=participant_folder)
    emg_model = pipeline_EMG_from_online(data['emg_emg'], data['label_emg'], num_classes=5, model_path=None, save=False, model_path_save = "models/model_ffnn_emg.pth", participant_folder=participant_folder)

# %% DATA ANALYSIS

metrics_df, confusion_matrices = data_analysis()
plot_accuracy_boxplots(metrics_df, 'FFNN')
plot_accuracy_boxplots(metrics_df, 'CNN')
plot_accuracy_boxplots(metrics_df, 'LSTM')
plot_performance_lineplot(metrics_df, 'FFNN')
plot_performance_lineplot(metrics_df, 'CNN')
plot_performance_lineplot(metrics_df, 'LSTM')
plot_f1_scores_barchart(metrics_df, 'FFNN')
plot_f1_scores_barchart(metrics_df, 'CNN')
plot_f1_scores_barchart(metrics_df, 'LSTM')

data = comparison_analysis(metrics_df)
# Pivot the data to create a matrix of Classification_Method vs Input_Modality with Accuracy values
accuracy_matrix = data.pivot_table(values='Accuracy', 
                                   index='Classification_Method', 
                                   columns='Input_Modality')
# Call the function
plot_heatmap(accuracy_matrix)
# Call the function
bar_chart_with_error_bars(data)
# %%

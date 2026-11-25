#%%
import rclpy
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64, Int32, Bool, String
from threading import Thread, Lock
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.transform import Rotation
from scipy import signal
from collections import Counter
from .preprocessing_and_normalization import notch_filter, bandpass_filter, highpass_filter, convert_to_SI, normalize_raw_imu, normalize_EMG_all_channels, undersample_majority_class_first_n
from .model import MyMultimodalNetwork, get_online_tensor_dataset, train_multiclass, MyMultimodalNetworkLSTM, train_lstm, evaluate_lstm, inference_lstm, MyMultimodalNetworkCNN, train_cnn, evaluate_cnn, inference_cnn, MyEMGNetwork, train_EMG, test_EMG, MyNetworkLSTM, train_EMG_lstm, evaluate_EMG_lstm, MyNetworkCNN, train_EMG_cnn, evaluate_EMG_cnn
from .feature_extraction import get_online_feature_vector, extract_features
# my custom msg
from interfaces.msg import RecordingFolder
import os

#%% DATA MANAGER CLASS

# class for storing the input vectors, true and predicted labels during the recording mode and store them in the participant folder
class DataManagerRawImu:
    def __init__(self):
        self.buffer = []
        self.recording_time_start = 0.0

    def append_data(self, emg, imu, label, prediction1, prediction2, prediction3):
        print(f"PREDICTIONS: {prediction1}, {prediction2}, {prediction3}")
        emg = emg.squeeze()
        if isinstance(imu, str) and imu == '0':
            imu = np.zeros_like(emg)  # or any appropriate placeholder
        
        batch_results = np.array([
            emg.cpu().numpy(),
            imu.cpu().numpy() if not isinstance(imu, str) else imu,
            label.cpu().numpy(),
            prediction1,
            prediction2,
            prediction3
        ], dtype=object)
        
        self.buffer.append(batch_results)
        print(f"len of the buffer raw imu: {len(self.buffer)}")

    def save_data(self, directory=None):
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        file_name = "raw_imu_dataset.npz"
        file_path = os.path.join(directory, file_name) if directory else file_name
        
        if len(self.buffer) == 0:
            print("\n\n\n\n\n\n\n PASS")
            return
        
        data_dict = {
            'emg': np.array([item[0] for item in self.buffer]),
            'imu': np.array([item[1] for item in self.buffer]),
            'label': np.array([item[2] for item in self.buffer]),
            'prediction_0': np.array([item[3] for item in self.buffer]),
            'prediction_1': np.array([item[4] for item in self.buffer]),
            'prediction_2': np.array([item[5] for item in self.buffer])
        }
        
        np.savez(file_path, **data_dict)
        #np.savetxt(os.path.join(directory, 'recording_time_raw_imu.txt'), np.array([self.recording_time_start]), fmt='%f')
        self.reset_buffer()

    def load_data(self, file_name, directory=None):
        file_path = os.path.join(directory, file_name) if directory else file_name
        data = np.load(file_path, allow_pickle=True)
        return (data['emg'], data['imu'], data['label'], 
                data['prediction_0'], data['prediction_1'], data['prediction_2'])

    def save_recording_time(self, recording_time):
        self.recording_time_start = recording_time - self.recording_time_start

    def reset_buffer(self):
        self.buffer = []

#%% MAIN NODE

# global variables: state (mode) of the node, boolean for specify the first window of a movement (for calibration purpose), current label received from the GUI
work = 0
first_window = True
global_label = np.array([5]) # rest as default label
num_emg_channels = 9

class PipelineNodeRawImu(Node):
    def __init__(self):
        super().__init__('pipeline_node_raw_imu')
        # global variable for manage the different staes of the node
        global work
        # publish the inference to update the GUI
        self.inference_pub = self.create_publisher(Int32, '/inference32', qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # Used for notice the GUI that the training process is finished and come back to the not-working mode
        self.bool_pub = self.create_publisher(Bool, '/bool', qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # publisher for updating the score at each correct prediction
        self.score_pub = self.create_publisher(Bool, '/score', qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # change modes from GUI commands
        self.mode_sub = self.create_subscription(Int32, '/mode32', self.mode_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # For the calibration of the IMU sensors, each time there's a new label
        self.label_sub = self.create_subscription(Int32, '/label32', self.label_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        
        # subscriber for the online effective sampling frequency to send to the vqf 
        self.freq_sub = self.create_subscription(Float64, '/freq64', self.freq_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # variable for storing the effective sampling frequency
        self.freq = 0.0
        
        # subscriber for receiving participant informations in recording mode
        self.subscription = self.create_subscription(RecordingFolder, '/folder_info_raw_imu', self.participant_info_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # variable for storing the participant folder
        self.current_folder = None
        # Training path subscription
        self.training_path_sub = self.create_subscription(String, '/training_path', self.training_path_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # variable for storing the training path folder
        self.training_path = None        
        # subscriber to get the window from acquisition node
        self.window_sub = self.create_subscription(Float64MultiArray, '/window32', self.window_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # flag for not executing pipeline after changing the label for an amount of time
        self.sleeping_flag = True
        # data manager for storing the data during recording mode and use it for training
        self.data_manager = DataManagerRawImu()
        # Neural Network
        self.config = {
            'model_type': None, # 'ffnn', 'lstm', 'cnn'
            'input_type': None, # 'emg', 'emg+raw_imu', 'emg+angles'
            'num_classes': 5,
            'hidden_sizes_emg': None, #[128, 128, 128],[512, 1024, 1024, 1024, 512],
            'hidden_sizes_imu': None, #[128, 128, 128],#[512, 1024, 1024, 1024, 512],
            'input_shape_emg': None,
            'input_shape_imu': None, #9,
            'dropout_rate': 0.1
        }
        self.train_config = {
            "batch_size": 1,
            "epochs": 20,
            "optimizer": "sgd",
            "learning_rate": 0.1
        }
        self.channel_max = np.load('max_emg.npy')
        self.mean_raw_imu = None #np.load('mean_raw_imu.npy')
        self.std_raw_imu = None #np.load('std_raw_imu.npy')
        self.models = [self.model_ffnn_raw_imu, self.model_lstn_raw_imu, self.model_cnn_raw_imu] = [None] * 3
        self.models_type = ['ffnn', 'lstm', 'cnn']
        self.config['input_type'] = 'emg+raw_imu'
        self.predictions = [None] * 3
        self.logits = [None] * 3
        self.model = None
        for i in range(3):
            self.config['model_type'] = self.models_type[i]
            self.models[i] = self.create_and_load_model()

        # only for plotting angles
        self.angles_plot = np.zeros((9,1))

    def create_and_load_model(self, return_config=False):
        global num_emg_channels
        if self.config['model_type'] == 'ffnn':
            if self.config['input_type'] == 'emg+raw_imu':
                self.mean_raw_imu = np.load('mean_raw_imu.npy')
                self.std_raw_imu = np.load('std_raw_imu.npy')
                self.config['input_shape_emg'] = num_emg_channels*4
                self.config['input_shape_imu'] = 18
                self.config['hidden_sizes_imu'] = [256, 256, 256]
                self.config['hidden_sizes_emg'] = [256, 256, 256]
                self.model = MyMultimodalNetwork(input_shape_emg=self.config['input_shape_emg'], input_shape_imu=self.config['input_shape_imu'], 
                                                 num_classes=self.config['num_classes'], hidden_sizes_emg=self.config['hidden_sizes_emg'], 
                                                 hidden_sizes_imu=self.config['hidden_sizes_imu'], dropout_rate=self.config['dropout_rate'])
                self.criterion = nn.CrossEntropyLoss()
                self.train_config['optimizer'] = 'adam'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.001
                #model_path = 'model_parameters_EMG_raw_imu.pth'
                model_path = 'models/model_ffnn_raw_imu.pth'
                if return_config:
                    return self.config, self.train_config
            elif self.config['input_type'] == 'emg+angles':
                self.config['input_shape_emg'] = (num_emg_channels, 4)
                self.config['input_shape_imu'] = 9
                self.config['hidden_sizes_imu'] = [256, 256, 256]
                self.config['hidden_sizes_emg'] = [256, 256, 256]
                self.model = MyMultimodalNetwork(input_shape_emg=self.config['input_shape_emg'], input_shape_imu=self.config['input_shape_imu'], 
                                                 num_classes=self.config['num_classes'], hidden_sizes_emg=self.config['hidden_sizes_emg'], 
                                                 hidden_sizes_imu=self.config['hidden_sizes_imu'], dropout_rate=self.config['dropout_rate'])
                self.criterion = nn.CrossEntropyLoss()
                self.train_config['optimizer'] = 'adam'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.001
                #model_path = 'model_parameters_EMG_angles.pth'
                model_path = 'models/model_ffnn.pth'
                if return_config:
                    return self.config, self.train_config
            elif self.config['input_type'] == 'emg':
                self.config['input_shape_emg'] = (num_emg_channels, 4)
                self.config['hidden_sizes_emg'] = [512, 512, 512]
                # Is not this one the network
                self.model = MyEMGNetwork(input_shape_emg=self.config['input_shape_emg'], num_classes=self.config['num_classes'], 
                                          hidden_sizes_emg=self.config['hidden_sizes_emg'])
                self.criterion = nn.CrossEntropyLoss()
                self.train_config['optimizer'] = 'sgd'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.1
                #model_path = 'model_parameters_EMG.pth'
                model_path = 'models/model_ffnn_emg.pth'
                if return_config:
                    return self.config, self.train_config
        elif self.config['model_type'] == 'lstm':
            if self.config['input_type'] == 'emg+raw_imu':
                self.mean_raw_imu = np.load('mean_raw_imu.npy')
                self.std_raw_imu = np.load('std_raw_imu.npy')
                self.config['input_shape_emg'] = num_emg_channels*4
                self.config['input_shape_imu'] = 18
                self.config['hidden_sizes_imu'] = [128, 64, 32]
                self.config['hidden_sizes_emg'] = [128, 64, 32]
                self.model = MyMultimodalNetworkLSTM(input_shape_emg=self.config['input_shape_emg'], input_shape_imu=self.config['input_shape_imu'], 
                                                     num_classes=self.config['num_classes'], hidden_sizes_emg=self.config['hidden_sizes_emg'], 
                                                     hidden_sizes_imu=self.config['hidden_sizes_imu'], dropout_rate=self.config['dropout_rate'], raw_imu = True)
                self.criterion = nn.BCEWithLogitsLoss()
                self.train_config['optimizer'] = 'adam'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.001
                #model_path = 'model_params_raw_imu_lstm.pth'
                model_path = 'models/model_lstm_raw_imu.pth'
                if return_config:
                    return self.config, self.train_config
            elif self.config['input_type'] == 'emg+angles':
                self.config['input_shape_emg'] = (num_emg_channels,4)
                self.config['input_shape_imu'] = 9
                self.config['hidden_sizes_imu'] = [128, 128, 128]
                self.config['hidden_sizes_emg'] = [128, 128, 128]
                self.model = MyMultimodalNetworkLSTM(input_shape_emg=self.config['input_shape_emg'], input_shape_imu=self.config['input_shape_imu'], 
                                                     num_classes=self.config['num_classes'], hidden_sizes_emg=self.config['hidden_sizes_emg'], 
                                                     hidden_sizes_imu=self.config['hidden_sizes_imu'], dropout_rate=self.config['dropout_rate'], squeeze=True)
                self.criterion = nn.BCEWithLogitsLoss()
                self.train_config['optimizer'] = 'adam'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.001
                #model_path = 'model_params_lstm.pth'
                model_path = 'models/model_lstm.pth'
                if return_config:
                    return self.config, self.train_config
            elif self.config['input_type'] == 'emg':
                self.config['input_shape_emg'] = (num_emg_channels, 4)
                self.config['hidden_sizes_emg'] = [256, 256, 256]
                # Is not this one the network
                self.model = MyNetworkLSTM(input_shape_emg=self.config['input_shape_emg'], num_classes=self.config['num_classes'], 
                                           hidden_sizes_emg=self.config['hidden_sizes_emg'], dropout_rate=self.config['dropout_rate'])
                self.criterion = nn.BCEWithLogitsLoss()
                self.train_config['optimizer'] = 'adam'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.001
                #model_path = 'model_params_emg_lstm.pth'
                model_path = 'models/model_lstm_emg.pth'
                if return_config:
                    return self.config, self.train_config
        elif self.config['model_type'] == 'cnn':
            if self.config['input_type'] == 'emg+raw_imu':
                self.mean_raw_imu = np.load('mean_raw_imu.npy')
                self.std_raw_imu = np.load('std_raw_imu.npy')
                self.config['input_shape_emg'] = num_emg_channels*4
                self.config['input_shape_imu'] = 18
                self.config['hidden_sizes_imu'] = [256, 128, 128]
                self.config['hidden_sizes_emg'] = [256, 128, 128]
                self.model = MyMultimodalNetworkCNN(input_shape_emg=self.config['input_shape_emg'], input_shape_imu=self.config['input_shape_imu'], 
                                                    num_classes=self.config['num_classes'], hidden_sizes_emg=self.config['hidden_sizes_emg'], 
                                                    hidden_sizes_imu=self.config['hidden_sizes_imu'], dropout_rate=self.config['dropout_rate'], raw_imu = True)
                self.criterion = nn.BCEWithLogitsLoss()
                self.train_config['optimizer'] = 'adam'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.001
                #model_path = 'model_params_raw_imu_cnn.pth'
                model_path = 'models/model_cnn_raw_imu.pth'
                if return_config:
                    return self.config, self.train_config
            elif self.config['input_type'] == 'emg+angles':
                self.config['input_shape_emg'] = (num_emg_channels, 4)
                self.config['input_shape_imu'] = 9
                self.config['hidden_sizes_imu'] = [128, 128, 128]
                self.config['hidden_sizes_emg'] = [128, 128, 128]
                self.model = MyMultimodalNetworkCNN(input_shape_emg=self.config['input_shape_emg'], input_shape_imu=self.config['input_shape_imu'], 
                                                    num_classes=self.config['num_classes'], hidden_sizes_emg=self.config['hidden_sizes_emg'], 
                                                    hidden_sizes_imu=self.config['hidden_sizes_imu'], dropout_rate=self.config['dropout_rate'])
                self.criterion = nn.CrossEntropyLoss()
                self.train_config['optimizer'] = 'adam'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.001
                #model_path = 'model_params_cnn.pth'
                model_path = 'models/model_cnn.pth'
                if return_config:
                    return self.config, self.train_config
            elif self.config['input_type'] == 'emg':
                self.config['input_shape_emg'] = (num_emg_channels, 4)
                self.config['hidden_sizes_emg'] = [256, 256, 256]
                self.model = MyNetworkCNN(input_shape_emg=self.config['input_shape_emg'], num_classes=self.config['num_classes'], 
                                          hidden_sizes_emg=self.config['hidden_sizes_emg'], dropout_rate=self.config['dropout_rate'])
                self.criterion = nn.CrossEntropyLoss()
                self.train_config['optimizer'] = 'sgd'
                self.optimizer = self.build_optimizer()
                self.train_config['learning_rate'] = 0.01
                #model_path = 'model_params_emg_cnn.pth'
                model_path = 'models/model_cnn_emg.pth'
                if return_config:
                    return self.config, self.train_config
        #print(model_path)
        self.model.load_state_dict(torch.load(model_path))
        return self.model

    # receive and manage the window for the pipeline
    def window_callback(self, msg):
        global global_label
        window = np.array(msg.data)
        label = global_label
        if self.sleeping_flag:
            self.pipeline_from_window(window, label)
        else:
            # If we start moving, we go on with the acquisition and pipeline
            # compute total acceleration of the wrist and compare it with a threshold
            acc_wrist = window[-3600 : -3600 + 600].reshape(3, 200)
            gyro_wrist = window[-3600 + 600 : -3600 + 1200].reshape(3, 200)
            # Calculate magnitudes
            acc_mag_wrist = np.linalg.norm(acc_wrist, axis=1)
            gyro_mag_wrist = np.linalg.norm(gyro_wrist, axis=1)
            # Check if movement is detected
            if (acc_mag_wrist > 50).any() or (gyro_mag_wrist > 50).any():
                self.sleeping_flag = True
                self.pipeline_from_window(window, label)

    
    def pipeline_from_window(self, window, label):
        global work
        global num_emg_channels
        if work == 1 or work == 3:
            window = np.vstack((window[:num_emg_channels*200].reshape(num_emg_channels, 200), window[-3600:].reshape(18, 200)))
            #print("received and processing")
            if self.config['input_type'] == "emg":
                window[:num_emg_channels,:] = notch_filter(bandpass_filter(highpass_filter(window[:num_emg_channels,:], 0.5), 0.5, 100), 50, 30)
                emg_vector = normalize_EMG_all_channels(extract_features(window[:num_emg_channels,:]), self.channel_max)
                label = self.one_hot_encoding(label)
                emg_data = np.array(emg_vector).reshape(1, num_emg_channels, 4)
                label_data = np.array(label).reshape(1, 5)
                assert len(emg_data) == len(label_data)
                emg_tensor = torch.tensor(emg_data, dtype=torch.float32)
                label_tensor = torch.tensor(label_data, dtype=torch.float32)
                loader = torch.utils.data.DataLoader(TensorDataset(emg_tensor, label_tensor), batch_size=1)
                if work == 1:
                    for i in range(3):
                        self.config['model_type'] = self.models_type[i]
                        self.model = self.models[i]
                        self.logits[i], self.predictions[i] = self.inference(loader)
                elif work == 3:
                    for i in range(3):
                        self.config['model_type'] = self.models_type[i]
                        self.model = self.models[i]
                        self.predictions[i] = self.inference_and_store(loader)
                    for batch in loader:
                        emg_tensor, label_tensor = [t.squeeze(0) for t in batch]
                        self.data_manager.append_data(emg_tensor, '0', label_tensor, *self.predictions)
                    print("data_appended")
            else:
                if self.config['input_type'] == "emg+angles":
                    emg_vector, imu_vector, label = self.pipeline(window, label)
                elif self.config['input_type'] == "emg+raw_imu":
                    emg_vector, imu_vector, label = self.pipeline_raw_imu(window, label)
                if work == 1:
                    tensor_dataset= get_online_tensor_dataset(emg_vector, imu_vector, label, self.config)
                    for i in range(3):
                        self.config['model_type'] = self.models_type[i]
                        if self.config['input_type'] == "emg+angles":
                            self.model = self.models[i]
                        elif self.config['input_type'] == "emg+raw_imu":
                            self.model = self.models[i]
                        loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=1)
                        self.logits[i], self.predictions[i] = self.inference(loader)
                    tensor_dataset = None
                elif work == 3:
                    tensor_dataset= get_online_tensor_dataset(emg_vector, imu_vector, label, self.config)
                    loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=1)
                    try:
                        for i in range(3):
                            self.config['model_type'] = self.models_type[i]
                            self.model = self.models[i]
                            self.predictions[i] = self.inference_and_store(loader)
                        # Process all batches
                        for batch in loader:
                            emg_tensor, imu_tensor, label_tensor = [t.squeeze(0) for t in batch]
                            self.data_manager.append_data(emg_tensor, imu_tensor, label_tensor, *self.predictions)
                        print("All data appended")
                    except Exception as e:
                        print(f"An error occurred during processing: {e}")
                    finally:
                        del tensor_dataset
                        del loader
    
    def pipeline(self, window, label):
        global first_window
        global num_emg_channels
        # (channels, samples) = (29, 200)
        window[:num_emg_channels,:] = notch_filter(bandpass_filter(highpass_filter(window[:num_emg_channels,:], 0.5), 0.5, 100), 50, 30)
        window = convert_to_SI(window)
        label = self.one_hot_encoding(label)
        emg_vector, imu_vector = get_online_feature_vector(window, label, first_window, self.freq, self.config['model_type'])
        emg_vector = normalize_EMG_all_channels(emg_vector, self.channel_max)
        first_window = False
        return emg_vector, imu_vector, label

    def pipeline_raw_imu(self, window, label):
        global first_window
        global num_emg_channels
        # (channels, samples) = (29, 200)
        window[:num_emg_channels,:] = notch_filter(bandpass_filter(highpass_filter(window[:num_emg_channels,:], 0.5), 0.5, 100), 50, 30)
        emg_vector = np.array(normalize_EMG_all_channels(extract_features(window[:num_emg_channels,:]), self.channel_max))
        window = convert_to_SI(window)
        imu_vector = normalize_raw_imu(window[num_emg_channels:,:], self.mean_raw_imu, self.std_raw_imu)
        imu_vector = np.mean(imu_vector, axis=1).squeeze()
        label = self.one_hot_encoding(label)
        return emg_vector, imu_vector, label
    
    def build_optimizer(self, optimizer=None):
        if optimizer is None:
            optimizer = self.train_config['optimizer']
        if optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.train_config['learning_rate'], momentum=0.9)
        elif optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.train_config['learning_rate'])
        return optimizer

    def one_hot_encoding(self, label):
        one_hot_matrix = np.zeros((1, self.config['num_classes']))
        if label >= 0 and label < self.config['num_classes']:
            one_hot_matrix[0, label] = 1.
        else:
            one_hot_matrix[0, 0] = 1.
        return one_hot_matrix

    def publish_data(self, data):
        msg = Int32()
        msg.data = data
        self.inference_pub.publish(msg)
    
    # when we enter the training mode
    def training_path_callback(self, msg):
        self.training_path = msg.data
        print(f"Received training path: {self.training_path}")
        last_char = self.training_path[-1]
        parts = self.training_path.split('/')
        parts[-1] = "DATASET_RAW_IMU_" + last_char
        self.training_path = '/'.join(parts)
        self.training_pipeline()
    
    def participant_info_callback(self, msg):
        self.current_folder = msg.folder_path
        print(f"Received participant info. Recording folder: {self.current_folder}")

    def save_participant_data(self):
        print("\n\n\n\n\n\n\n\n\n\n\ We save data in:")
        print(self.current_folder)
        self.data_manager.save_data(directory=self.current_folder)

    def mode_callback(self, msg):
        global work
        if msg.data == 4:
            self.save_participant_data()
            work = 1
        else:
            work = msg.data

    def label_callback(self, msg):
        global first_window
        global global_label
        first_window = True
        global_label = np.array([msg.data])
        self.sleeping_flag = False
        #time.sleep(1)
        #self.sleeping_flag = True

    def freq_callback(self, msg):
        self.freq = msg.data
        print(f"Effective sampling frequency: {self.freq}")

    def inference(self, loader):
        if self.config['input_type'] == 'emg':
            self.model.eval()
            with torch.no_grad():
                for emg, label in loader:
                    _, predicted = torch.max(self.model(emg), 1)
        else:
            self.model.eval()
            with torch.no_grad():
                for emg, imu, label in loader:
                    _, predicted = torch.max(self.model(emg, imu), 1)
        self.publish_data(int(np.array(predicted)))
        return np.array(predicted), np.array(_)

    def inference_and_store(self, loader):
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch in loader:
                if self.config['input_type'] == 'emg':
                    emg, label = batch
                    output = self.model(emg)
                else:
                    emg, imu, label = batch
                    output = self.model(emg, imu)
                
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
                
                self.publish_data(int(predicted))
                labels_idx = torch.argmax(label, dim=1)
                
                if predicted == labels_idx:
                    msg = Bool()
                    msg.data = True
                    self.score_pub.publish(msg)

        return np.array(all_predictions)
                
    def training_pipeline(self):
        print(self.training_path)
        emg_data, imu_data, label_data, prediction1, prediction2, prediction3 = self.data_manager.load_data('raw_imu_dataset.npz', directory=self.training_path)
        print(f"Emg data: {emg_data.shape}, IMU data: {imu_data.shape}, Labels: {label_data.shape}, Predictions: {prediction1.shape}{prediction2.shape}{prediction3.shape}")

        # select only the wrong ones
        # Convert label_data from one-hot encoding to class indices
        true_labels = torch.argmax(torch.tensor(label_data), dim=1)

        # Convert predictions to class indices
        pred1_indices = torch.tensor(prediction1).squeeze().long()
        pred2_indices = torch.tensor(prediction2).squeeze().long()
        pred3_indices = torch.tensor(prediction3).squeeze().long()

        # Identify incorrect predictions for each model
        incorrect_mask1 = pred1_indices != true_labels
        incorrect_mask2 = pred2_indices != true_labels
        incorrect_mask3 = pred3_indices != true_labels

        # Filter out the incorrect predictions for each model (keeping one-hot labels)
        wrong_emg_data1 = emg_data[incorrect_mask1]
        wrong_imu_data1 = imu_data[incorrect_mask1]
        wrong_label_data1 = label_data[incorrect_mask1]  # Keep one-hot encoded labels

        wrong_emg_data2 = emg_data[incorrect_mask2]
        wrong_imu_data2 = imu_data[incorrect_mask2]
        wrong_label_data2 = label_data[incorrect_mask2]  # Keep one-hot encoded labels

        wrong_emg_data3 = emg_data[incorrect_mask3]
        wrong_imu_data3 = imu_data[incorrect_mask3]
        wrong_label_data3 = label_data[incorrect_mask3]  # Keep one-hot encoded labels

        # Convert to tensors
        emg_tensor1 = torch.tensor(wrong_emg_data1, dtype=torch.float32)
        imu_tensor1 = torch.tensor(wrong_imu_data1, dtype=torch.float32)
        label_tensor1 = torch.tensor(wrong_label_data1, dtype=torch.float32)  # Still one-hot encoded

        emg_tensor2 = torch.tensor(wrong_emg_data2, dtype=torch.float32)
        imu_tensor2 = torch.tensor(wrong_imu_data2, dtype=torch.float32)
        label_tensor2 = torch.tensor(wrong_label_data2, dtype=torch.float32)  # Still one-hot encoded

        emg_tensor3 = torch.tensor(wrong_emg_data3, dtype=torch.float32)
        imu_tensor3 = torch.tensor(wrong_imu_data3, dtype=torch.float32)
        label_tensor3 = torch.tensor(wrong_label_data3, dtype=torch.float32)  # Still one-hot encoded

        # Create DataLoaders
        loader1 = DataLoader(TensorDataset(emg_tensor1, imu_tensor1, label_tensor1), batch_size=self.train_config['batch_size'], shuffle=True)
        loader2 = DataLoader(TensorDataset(emg_tensor2, imu_tensor2, label_tensor2), batch_size=self.train_config['batch_size'], shuffle=True)
        loader3 = DataLoader(TensorDataset(emg_tensor3, imu_tensor3, label_tensor3), batch_size=self.train_config['batch_size'], shuffle=True)
        loaders = [loader1, loader2, loader3]

        for i in range(3):
            self.config['model_type'] = self.models_type[i]
            self.model = self.models[i]
            print(f"\n TRAINING{i} is beginning:\n")
            train_multiclass(self.model, loaders[i], self.criterion, self.optimizer, self.train_config['epochs'])
            print("\n TRAINING is finished!\n")
            self.models[i] = self.model
        msg = Bool()
        msg.data = False
        self.bool_pub.publish(msg)
        #save the model
        for i in range(3):
            model_path = f'models/model_angles_{self.models_type[i]}.pth'
            torch.save(self.models[i].state_dict(), model_path)
        return

def spin_node(executor):
    executor.spin()

def main(args=None):
    rclpy.init(args=args)

    node = PipelineNodeRawImu()
    acq_executor = SingleThreadedExecutor()
    acq_executor.add_node(node)
    srv_thread = Thread(target=spin_node, args=(acq_executor,), daemon=True)
    srv_thread.start()

    try:
        srv_thread.join()  # Wait for the thread to finish
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')

    node.destroy_node()
    rclpy.shutdown()

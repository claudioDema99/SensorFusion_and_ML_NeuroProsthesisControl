#%%
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64, Int32, Bool, String
from threading import Thread, Lock
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
# my custom msg
import os

#%% DATA MANAGER CLASS

counter = 1

# class for storing the input vectors, true and predicted labels during the recording mode and store them in the participant folder
class DataManagerNormalization:
    def __init__(self):
        self.buffer = []

    def append_data(self, emg, imu):
        emg = emg.squeeze()
        if isinstance(imu, str) and imu == '0':
            imu = np.zeros_like(emg)  # or any appropriate placeholder
        print(f"Appending inside the data manager EMG: {emg.shape}, IMU: {imu.shape}")
        batch_results = np.array([
            emg,
            imu if not isinstance(imu, str) else imu,
        ], dtype=object)
        
        self.buffer.append(batch_results)
        print(f"len of the buffer: {len(self.buffer)}")

    def save_data(self, directory="raw_dataset"):
        global counter
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        file_name = f"raw_dataset_{counter}.npz"
        file_path = os.path.join(directory, file_name) if directory else file_name
        counter += 1
        
        if len(self.buffer) == 0:
            print("\n\n\n\n\n\n\n LEN BUFFER 0")
            return
        
        data_dict = {
            'emg': np.array([item[0] for item in self.buffer]),
            'imu': np.array([item[1] for item in self.buffer]),
        }
        
        print(f"Saving data to {file_path}")
        try:
            np.savez(file_path, **data_dict)
        except Exception as e:
            print(f"Error saving data: {e}")
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = []

#%% MAIN NODE

# global variables: state (mode) of the node, boolean for specify the first window of a movement (for calibration purpose), current label received from the GUI
work = 0

class StoringRaw(Node):
    def __init__(self):
        super().__init__('storing_node')
        # global variable for manage the different staes of the node
        global work
        # change modes from GUI commands
        self.mode_sub = self.create_subscription(Int32, '/mode32', self.mode_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        # data manager for storing the data during recording mode and use it for training
        self.data_manager = DataManagerNormalization()
        self.callback_group = ReentrantCallbackGroup()
        # two subscribers for receiving emg and imu data from matlab
        self.emg_sub = self.create_subscription(Float64MultiArray, '/float64emg', self.emg_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=self.callback_group)
        self.imu_sub = self.create_subscription(Float64MultiArray, '/float64imu', self.imu_callback, qos_profile=rclpy.qos.qos_profile_sensor_data, callback_group=self.callback_group)

    def mode_callback(self, msg):
        global work
        global matlab_recording_time
        if msg.data == 4:
            print("Saving data")
            self.data_manager.save_data()
            work = 1
        else:
            work = msg.data

    def emg_callback(self, msg):
        self.storing(emg=np.array(msg.data))

    def imu_callback(self, msg):
        self.storing(imu=np.array(msg.data))
    
    def storing(self, emg=None, imu=None):
        global work
        # Initialize the buffer attributes on the first call
        if not hasattr(self, 'emg_list'):
            self.emg_list = []
        if not hasattr(self, 'imu_list'):
            self.imu_list = []
        if work == 3:
            if emg is not None:
                self.emg_list.append(emg)
            else:
                self.imu_list.append(imu)

            while len(self.emg_list) >= 1 and len(self.imu_list) >= 1:
                emg = self.emg_list.pop(0)
                imu = self.imu_list.pop(0)
                self.data_manager.append_data(emg, imu)
            
            if(len(self.emg_list) - len(self.imu_list) >= 10):
                self.emg_list = self.emg_list[10:]

def main(args=None):
    rclpy.init(args=args)
    
    node = StoringRaw()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        node.destroy_node()
        rclpy.shutdown()

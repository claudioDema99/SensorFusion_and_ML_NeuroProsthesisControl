# CBPR_Master_Thesis

## **Sensor Fusion for Enhancing Prosthesis Control: Overcoming the Limb Position Effect and Dynamic Movement Variability Using EMG and IMU Data (Neuro-Prosthesis)**

This project addresses the **limb position effect** in neuro-prosthesis control using EMG signals, through a **sensor-fusion algorithm** that integrates and processes **IMU data** from three sensors placed on the **wrist**, **forearm**, and **upper arm**.
I developed a sensor fusion and machine learning algorithm to enhance the control of transradial neural prostheses during daily life activities. Specifically, I worked on a pattern recognition algorithm capable of recognizing different objects the prosthesis is grasping, based on their shape and weight, independently of arm movement and position. This included integrating an orientation estimation algorithm to determine the arm’s position in space. I conducted a study with 15 volunteer participants to evaluate the algorithm’s performance, achieving promising results that are expected to be published and presented at RehabWeek 2025 in Chicago, IL, USA.

### **Project Goal**

The overall goal of the algorithm is to enhance **neuro-prosthesis control** during daily activities involving **object grasping** and **movement**. It achieves this by training **machine learning models** to recognize different **grasp gestures** and the **weight of objects**, independent of the arm's movement and position in space.

### **Angle Estimation Algorithm**

I developed an **angle estimation algorithm** that uses data from the **accelerometers** and **gyroscopes** to extract **9 angles** that accurately define the position of the arm in space. This is achieved using an **IMU orientation estimation with bias estimation algorithm** applied to the raw IMU data.

### **Machine Learning Architectures**

Three supervised learning architectures were tested:

- **FFNN (Feed-Forward Neural Network)**
- **CNN (Convolutional Neural Network)**
- **LSTM (Long Short-Term Memory Network)**

Each architecture was tested using three different input vectors:

1. **Features extracted from EMG data**
2. **EMG features combined with raw IMU data**
3. **EMG features combined with arm angles estimated from IMU data**

### **Performance & Results**

The **angle estimation algorithm** demonstrated **significant improvements** and **better performance** across all machine learning architectures. These results were validated through **offline experiments** with **15 volunteer participants**, providing a strong foundation for potential **future online testing**. ToDo

### Pipeline Scripts

The scripts **`Pipeline`**, **`Pipeline_CNN`**, and **`Pipeline_LSTM`** are responsible for managing the entire **offline** process for each model. This includes loading datasets, preprocessing, feature extraction, inference, data analysis, and plotting. The models utilized in these scripts are found in the **`cbpr_master_thesis`** folder, specifically in the submodules: `data_analysis`, `feature_extraction`, `model`, and `preprocessing_and_normalization`. The script **`Pipeline_hyperparams`** is used for hyperparameter tuning across the various models.

### MATLAB Code

The **MATLAB** folder contains the primary MATLAB code, which is based on the **LocoD platform** with necessary modifications. This code is used for managing the acquisition of signals from the **Delsys system**, storing datasets, and streaming data to the ROS2 environment through the **MATLAB ROS2 Toolbox**.

### Params Folder

The **params** folder is used to store the parameters required for normalizing EMG and raw IMU data specific to each participant.

### ROS2 Folder and Infrastructure

The **ROS2** folder includes the packages for managing real-time streaming, pipelines, and inference of all models within the ROS2 environment. It also handles the storage of input vectors, predictions, and provides the ability to fine-tune models based on the stored dataset. These processes are efficiently controlled through a GUI that interacts with all the nodes, allowing users to manage the entire infrastructure.

### How to Run the ROS2 Infrastructure

To run the ROS2 infrastructure:

1. Ensure **ROS2** is installed on your Linux environment or WSL (Windows).
2. Download the **ROS2** folder, and place the two packages (**delsys_pkg** and **interfaces**) into your ROS2 workspace.
3. Open the terminal and navigate to your ROS2 workspace folder.
4. Build the selected packages using `colcon build --packages-select delsys_pkg interfaces`.
5. Run the infrastructure with `ros2 launch delsys_pkg delsys_launch.py`.

### How to Use the System

#### Prerequisites

- You need a **Delsys Trigno System** (Avanti and EMG sensors).
  
#### Steps

1. Connect the **Delsys Trigno** to your PC.
2. Run the ROS2 infrastructure.
3. Open the **participant panel** and create a participant folder in the filesystem.
4. In non-working mode, current predictions can be viewed by switching to working mode.
5. For recording sessions, select "go recording" to record data, which will then be saved inside the participant's folder under the recording session number.
6. Once at least one recording session is saved, models can be trained in training mode. A slicing panel will allow you to select which recording session to use for training.
ToDo

For further details on the motivations, background, decisions, and logic behind the project, refer to the full **Master Thesis**.

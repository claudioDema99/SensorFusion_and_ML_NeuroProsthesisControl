# CBPR_Master_Thesis

## Sensor Fusion for Enhancing Prosthesis Control: Overcoming the Limb Position Effect and Dynamic Movement Variability Using EMG and IMU Data (Neuro-Prosthesis)

This project addresses the **limb position effect** in neuro-prosthesis control using EMG signals, through a sensor-fusion algorithm that integrates and processes IMU data from three sensors placed on the wrist, forearm, and upper arm.

The overall goal of the algorithm is to enhance neuro-prosthesis control during daily activities involving object grasping and movement. It does so by training machine learning models to recognize different grasp gestures and the weight of the objects, independent of the arm's movement and position in space.

I developed an angle estimation algorithm that uses data from the accelerometers and gyroscopes to extract 9 angles that accurately define the position of the arm in space. This is achieved with an IMU orientation estimation with bias estimation algorithm applied to the raw IMU data.

Three supervised learning architectures (FFNN, CNN, and LSTM) were tested, each using three different input vectors: features extracted from EMG data, EMG features combined with raw IMU data, and EMG features combined with arm angles estimated from IMU data.

The angle estimation algorithm demonstrated significant improvements and better performance across all machine learning architectures, as shown in offline experiments with 15 volunteer participants. These results provide a strong basis for potential future online testing.

The scripts 'Pipeline', 'Pipeline_CNN', and 'Pipeline_LSTM' are responsible for managing all the offline part of each model from the loading of the datasets, preprocessing, feature extraction, and inference, to the data analysis and plotting. The models used by them are in cbpr_master_thesis (data_analysis, feature_extraction, model, preprocessing_and_normalization)

The folder MATLAB contains the main Matlab code (from LocoD platform) used for managing the acquisition of the signal from Delsys system, storing of dataset and streaming to ROS2 environment thanks to MATLAB ROS2 TOOLBOX.

The ROS2 folder contains the packages used in ROS2 environment for managing real-time streaming, pipelines and inference of all the models, as well as storing of input vectors, predicitons, and the possibile to fine-tune the models on the stored dataset.
All this behaviours are managed properly by a GUI that interacts with all the nodes and let you to control the overall infrastructure.

HOW TO RUN THE ROS2 INFRASTRUCTURE:
YOU NEED ROS2 INSTALLED ON YOUR LINUX ENVIRONMENT, OR ON YOUR WSL (WINDOWS)
YOU NEED TO DOWNLOAD THE ROS2 FOLDER, AND PUT THE TWO PACKAGES (DELSYS_PKG AND INTERFACE) INTO YOUR ROS2 WORKSPACE
OPEN THE TERMINAL AND NAVIGATE TO YOUR ROS2 WORKSPACE FOLDER
BUILD WITH 'COLCON BUILD -P ..' THE PACKAGES SELECTED
RUN 'ROS2 LUNCH ..'

HOW TO USE IT:
PREREQUISITES:
YOU NEED DELSYS TRIGNO SYSTEM (AVANTI AND EMG SENSORS)
THEN:
CONNECT DELSYS TRIGNO TO YOUR PC (LINK)
RUN THE ROS2 INFRASTRUCTURE
ONCE OPENED, PARTICIPANT PANEL (AND CREATE PARTICIPANT FOLDER INSIDE FILESYSTEM)
THEN AUTOMATICALLY IN NOT-WORKING MODE
IF GO WORKING CAN SEE CURRENT PREDICTIONS
IF GO RECORDING RECORDS AND THEN SAVE INSIDE PARTICIPANT FOLDER / RECORDING SESSION NUMBER
ONCE YOU HAVE AT LEAST ONE RECORDING SESSION SAVED INSIDE YOUR PARTICIPANT FOLDER, YOU CAN TRAIN THE MODELS IN THE TRAINING MODE (A SLICING PANEL WILL ALLOW YOU TO SELECT THE RECORDING SESSION TO USE FOR TRAINING)

FOR ALL THE INFORMATION ABOUT THE MOTIVATIONS, BACKGROUND, DECISIONS AND LOGIC OF THE PROJECT: LINK TO MASTER THESIS
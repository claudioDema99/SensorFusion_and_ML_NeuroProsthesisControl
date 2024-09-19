# CBPR_Master_Thesis

## Sensor Fusion for Enhancing Prosthesis Control: Overcoming the Limb Position Effect and Dynamic Movement Variability Using EMG and IMU Data (Neuro-Prosthesis)

This project addresses the **limb position effect** in neuro-prosthesis control using EMG signals, through a sensor-fusion algorithm that integrates and processes IMU data from three sensors placed on the wrist, forearm, and upper arm.

The overall goal of the algorithm is to enhance neuro-prosthesis control during daily activities involving object grasping and movement. It does so by training machine learning models to recognize different grasp gestures and the weight of the objects, independent of the arm's movement and position in space.

I developed an angle estimation algorithm that uses data from the accelerometers and gyroscopes to extract 9 angles that accurately define the position of the arm in space. This is achieved with an IMU orientation estimation with bias estimation algorithm applied to the raw IMU data.

Three supervised learning architectures (FFNN, CNN, and LSTM) were tested, each using three different input vectors: features extracted from EMG data, EMG features combined with raw IMU data, and EMG features combined with arm angles estimated from IMU data.

The angle estimation algorithm demonstrated significant improvements and better performance across all machine learning architectures, as shown in offline experiments with 15 volunteer participants. These results provide a strong basis for potential future online testing.
#%%
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import tkinter as tk
from tkinter import font as tkfont, simpledialog
from PIL import Image, ImageTk, ImageEnhance
# my custom msg
from interfaces.msg import RecordingFolder
import os

BASE_DIR = "/home/dema/CBPR_Recording_Folders"
number_recordings = 1
training_path = ""

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

NOT_WORKING_MSG = "\nNOT-WORKING\n       MODE"
WORKING_MSG = "\nINFERENCE\n     MODE"
TRAINING_MSG = "\nTRAINING\n   MODE"
RECORDING_MSG = "\n\n\n\n\n\n\n\n\nRECORDING\n     MODE"
MSGS_LIST = [NOT_WORKING_MSG, WORKING_MSG, TRAINING_MSG, RECORDING_MSG]

#%% BUTTON CLASS

class ToggleButton(tk.Canvas):
    def __init__(self, parent, width, height, label, command=None):
        tk.Canvas.__init__(self, parent, width=width, height=height, bd=0, highlightthickness=0)
        self.command = command

        self.width = width
        self.height = height
        self.label = label

        self.switch_on = False
        self.enabled = True

        self.rect = self.create_rectangle(0, 0, width, height, fill="red", outline="")
        self.text = self.create_text(width//2, height//2, text=label, fill="white", font=("Arial", 10, "bold"))

        self.bind("<Button-1>", self.toggle)

    def toggle(self, event=None):
        if self.enabled:
            self.switch_on = not self.switch_on
            self.draw()
            if self.command is not None:
                self.command(self.switch_on)

    def draw(self):
        if self.switch_on:
            self.itemconfig(self.rect, fill="green")
        else:
            self.itemconfig(self.rect, fill="red")

    def set_state(self, state):
        self.switch_on = state
        self.draw()

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.itemconfig(self.rect, fill="gray" if not enabled else ("green" if self.switch_on else "red"))

#%% RECORDING MODE - INFO PANEL CLASS

class ParticipantInfoDialog(tk.Toplevel):
    def __init__(self, parent, param1, param2, param3):
        super().__init__(parent)
        self.title("Model Configuration Information")
        self.geometry("900x1050")
        self.configure(bg="#f0f0f0")

        # Custom fonts
        title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        label_font = tkfont.Font(family="Helvetica", size=12)
        button_font = tkfont.Font(family="Helvetica", size=12, weight="bold")

        self.model_type = param1[0]
        self.input_type = param1[1]
        self.num_classes = param1[2]
        self.hidden_sizes_emg = param1[3]
        self.hidden_sizes_imu = param1[4]
        self.input_shape_emg = param1[5]
        self.input_shape_imu = param1[6]
        self.dropout_rate = param1[7]
        self.batch_size = param1[8]
        self.epochs = param1[9]
        self.optimizer = param1[10]
        self.learning_rate = param1[11]

        self.model_type_emg = param2[0]
        self.input_type_emg = param2[1]
        self.num_classes_emg = param2[2]
        self.hidden_sizes_emg_emg = param2[3]
        self.hidden_sizes_imu_emg = param2[4]
        self.input_shape_emg_emg = param2[5]
        self.input_shape_imu_emg = param2[6]
        self.dropout_rate_emg = param2[7]
        self.batch_size_emg = param2[8]
        self.epochs_emg = param2[9]
        self.optimizer_emg = param2[10]
        self.learning_rate_emg = param2[11]

        self.model_type_raw_imu = param3[0]
        self.input_type_raw_imu = param3[1]
        self.num_classes_raw_imu = param3[2]
        self.hidden_sizes_emg_raw_imu = param3[3]
        self.hidden_sizes_imu_raw_imu = param3[4]
        self.input_shape_emg_raw_imu = param3[5]
        self.input_shape_imu_raw_imu = param3[6]
        self.dropout_rate_raw_imu = param3[7]
        self.batch_size_raw_imu = param3[8]
        self.epochs_raw_imu = param3[9]
        self.optimizer_raw_imu = param3[10]
        self.learning_rate_raw_imu = param3[11]

        # Title
        tk.Label(self, text="Models Configuration", font=title_font, bg="#f0f0f0").pack(pady=20)

        # Frame for form fields
        form_frame = tk.Frame(self, bg="#f0f0f0")
        form_frame.pack(pady=10)

        # Display model configuration
        tk.Label(form_frame, text=f"Model Type: {self.model_type}", font=label_font, bg="#f0f0f0").grid(row=0, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Type: {self.input_type}", font=label_font, bg="#f0f0f0").grid(row=1, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Hidden Sizes EMG: {self.hidden_sizes_emg}", font=label_font, bg="#f0f0f0").grid(row=2, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Hidden Sizes IMU: {self.hidden_sizes_imu}", font=label_font, bg="#f0f0f0").grid(row=3, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Shape EMG: {self.input_shape_emg}", font=label_font, bg="#f0f0f0").grid(row=4, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Shape IMU: {self.input_shape_imu}\n", font=label_font, bg="#f0f0f0").grid(row=5, column=0, sticky=tk.W, pady=5)

        tk.Label(form_frame, text=f"Model Type: {self.model_type_emg}", font=label_font, bg="#f0f0f0").grid(row=7, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Type: {self.input_type_emg}", font=label_font, bg="#f0f0f0").grid(row=8, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Hidden Sizes EMG: {self.hidden_sizes_emg_emg}", font=label_font, bg="#f0f0f0").grid(row=9, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Hidden Sizes IMU: {self.hidden_sizes_imu_emg}", font=label_font, bg="#f0f0f0").grid(row=10, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Shape EMG: {self.input_shape_emg_emg}", font=label_font, bg="#f0f0f0").grid(row=11, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Shape IMU: {self.input_shape_imu_emg}\n", font=label_font, bg="#f0f0f0").grid(row=12, column=0, sticky=tk.W, pady=5)

        tk.Label(form_frame, text=f"Model Type: {self.model_type_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=14, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Type: {self.input_type_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=15, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Hidden Sizes EMG: {self.hidden_sizes_emg_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=16, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Hidden Sizes IMU: {self.hidden_sizes_imu_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=17, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Shape EMG: {self.input_shape_emg_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=18, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Input Shape IMU: {self.input_shape_imu_raw_imu}\n", font=label_font, bg="#f0f0f0").grid(row=19, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Training and configuration parameters for all models:", font=label_font, bg="#f0f0f0").grid(row=21, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Dropout Rate: {self.dropout_rate_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=22, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Number of Classes: {self.num_classes}", font=label_font, bg="#f0f0f0").grid(row=23, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Batch Size: {self.batch_size_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=24, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Epochs: {self.epochs_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=25, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Optimizer: {self.optimizer_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=26, column=0, sticky=tk.W, pady=5)
        tk.Label(form_frame, text=f"Learning Rate: {self.learning_rate_raw_imu}", font=label_font, bg="#f0f0f0").grid(row=27, column=0, sticky=tk.W, pady=5)

        # Init button
        self.init_button = tk.Button(self, text="START RECORDING", command=self.on_init, 
                                     font=button_font, bg="#4CAF50", fg="white",
                                     activebackground="#45a049", activeforeground="white")
        self.init_button.pack(pady=20)

    def on_init(self):
        global BASE_DIR
        global number_recordings
        # Save model configuration info to a txt file
        participant_id = f"MODELS_INFO_{number_recordings}"  # Replace with the actual participant ID if needed
        file_path = os.path.join(BASE_DIR, f"{participant_id}.txt")
        number_recordings += 1
        
        with open(file_path, 'w') as f:
            f.write(f"\n Models:\n") # UNO DIETRO L'ALTRO DA SISTEMARE
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Input Type: {self.input_type}\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
            f.write(f"Hidden Sizes EMG: {self.hidden_sizes_emg}\n")
            f.write(f"Hidden Sizes IMU: {self.hidden_sizes_imu}\n")
            f.write(f"Input Shape EMG: {self.input_shape_emg}\n")
            f.write(f"Input Shape IMU: {self.input_shape_imu}\n")
            f.write(f"Dropout Rate: {self.dropout_rate}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Optimizer: {self.optimizer}\n")
            f.write(f"Learning Rate: {self.learning_rate}\n")
            f.write(f"\n\n")
            f.write(f"Model Type EMG: {self.model_type_emg}\n")
            f.write(f"Input Type EMG: {self.input_type_emg}\n")
            f.write(f"Number of Classes EMG: {self.num_classes_emg}\n")
            f.write(f"Hidden Sizes EMG EMG: {self.hidden_sizes_emg_emg}\n")
            f.write(f"Hidden Sizes IMU EMG: {self.hidden_sizes_imu_emg}\n")
            f.write(f"Input Shape EMG EMG: {self.input_shape_emg_emg}\n")
            f.write(f"Input Shape IMU EMG: {self.input_shape_imu_emg}\n")
            f.write(f"Dropout Rate EMG: {self.dropout_rate_emg}\n")
            f.write(f"Batch Size EMG: {self.batch_size_emg}\n")
            f.write(f"Epochs EMG: {self.epochs_emg}\n")
            f.write(f"Optimizer EMG: {self.optimizer_emg}\n")
            f.write(f"Learning Rate EMG: {self.learning_rate_emg}\n")
            f.write(f"\n\n")
            f.write(f"Model Type RAW IMU: {self.model_type_raw_imu}\n")
            f.write(f"Input Type RAW IMU: {self.input_type_raw_imu}\n")
            f.write(f"Number of Classes RAW IMU: {self.num_classes_raw_imu}\n")
            f.write(f"Hidden Sizes EMG RAW IMU: {self.hidden_sizes_emg_raw_imu}\n")
            f.write(f"Hidden Sizes IMU RAW IMU: {self.hidden_sizes_imu_raw_imu}\n")
            f.write(f"Input Shape EMG RAW IMU: {self.input_shape_emg_raw_imu}\n")
            f.write(f"Input Shape IMU RAW IMU: {self.input_shape_imu_raw_imu}\n")
            f.write(f"Dropout Rate RAW IMU: {self.dropout_rate_raw_imu}\n")
            f.write(f"Batch Size RAW IMU: {self.batch_size_raw_imu}\n")
            f.write(f"Epochs RAW IMU: {self.epochs_raw_imu}\n")
            f.write(f"Optimizer RAW IMU: {self.optimizer_raw_imu}\n")
            f.write(f"Learning Rate RAW IMU: {self.learning_rate_raw_imu}\n")
        self.destroy()

class ParticipantInfoDialogAfterInit(simpledialog.Dialog):
    def __init__(self, parent, title=None):
        super().__init__(parent, title)

    def body(self, master):
        # Set larger font for labels and entries
        label_font = ('Arial', 14)
        entry_font = ('Arial', 14)

        # Labels with larger font
        tk.Label(master, text="Participant ID:", font=label_font).grid(row=0, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Repository:", font=label_font).grid(row=1, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Name:", font=label_font).grid(row=2, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Gender (Male/Female/Other):", font=label_font).grid(row=3, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Age:", font=label_font).grid(row=4, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Height (cm):", font=label_font).grid(row=5, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Weight (kg):", font=label_font).grid(row=6, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Upper Arm Length (cm):", font=label_font).grid(row=7, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Forearm Length (cm):", font=label_font).grid(row=8, column=0, sticky="e", padx=10, pady=5)

        # Entries with larger font
        self.participant_id = tk.Entry(master, font=entry_font, width=30)
        self.repository = tk.Entry(master, font=entry_font, width=30)
        self.name = tk.Entry(master, font=entry_font, width=30)
        self.gender = tk.Entry(master, font=entry_font, width=30)
        self.age = tk.Entry(master, font=entry_font, width=30)
        self.height = tk.Entry(master, font=entry_font, width=30)
        self.weight = tk.Entry(master, font=entry_font, width=30)
        self.upper_arm_length = tk.Entry(master, font=entry_font, width=30)
        self.forearm_length = tk.Entry(master, font=entry_font, width=30)

        # Grid positions with added padding
        self.participant_id.grid(row=0, column=1, padx=10, pady=5)
        self.repository.grid(row=1, column=1, padx=10, pady=5)
        self.name.grid(row=2, column=1, padx=10, pady=5)
        self.gender.grid(row=3, column=1, padx=10, pady=5)
        self.age.grid(row=4, column=1, padx=10, pady=5)
        self.height.grid(row=5, column=1, padx=10, pady=5)
        self.weight.grid(row=6, column=1, padx=10, pady=5)
        self.upper_arm_length.grid(row=7, column=1, padx=10, pady=5)
        self.forearm_length.grid(row=8, column=1, padx=10, pady=5)

        # Increase the size of the dialog window
        self.geometry("600x400")

        return self.participant_id  # Set focus to the first entry field

    def geometry(self, size):
        self.master.geometry(size)

    def apply(self):
        # Get values from entries
        self.participant_id = self.participant_id.get()
        self.repository = self.repository.get()
        self.name = self.name.get()
        self.gender = self.gender.get()
        self.age = self.age.get()
        self.height = self.height.get()
        self.weight = self.weight.get()
        self.upper_arm_length = self.upper_arm_length.get()
        self.forearm_length = self.forearm_length.get()

#%% TRAINING MODE - FINE TUNING DATA SELECTION PANEL CLASS

class TrainingPanel(tk.Toplevel, Node):
    def __init__(self, parent, path):
        super().__init__(parent)
        self.title("Select the recording folder for fine-tuning")
        self.geometry("600x500")  # Increased height to accommodate new fields
        self.configure(bg="#f0f0f0")

        # Create a listbox to display subfolders
        self.subfolder_listbox = tk.Listbox(self, selectmode=tk.SINGLE, bg="white", font=("Arial", 12))
        self.subfolder_listbox.pack(pady=20, fill=tk.BOTH, expand=True)

        # Add a scroll bar
        scrollbar = tk.Scrollbar(self.subfolder_listbox, orient="vertical")
        scrollbar.config(command=self.subfolder_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.subfolder_listbox.config(yscrollcommand=scrollbar.set)

        entries = os.listdir(path)
        subfolders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        # Populate the listbox with subfolders
        for subfolder in subfolders:
            self.subfolder_listbox.insert(tk.END, subfolder)

        # Add a select button
        select_button = tk.Button(self, text="Select Subfolder", command=self.return_selected_folder, bg="#4CAF50", fg="white", font=("Arial", 12))
        select_button.pack(pady=10)
    
    def return_selected_folder(self):
        global training_path
        selected_index = self.subfolder_listbox.curselection()

        if selected_index:
            selected_folder = self.subfolder_listbox.get(selected_index)
            training_path = selected_folder
            print(f"Selected folder: {selected_folder}")
            self.destroy()  # Close the panel


#%% MAIN GUI

class MultiObjectGUI(Node):
    def __init__(self):
        super().__init__('multi_object_gui')
        # to get the inference output from the network
        self.subscription = self.create_subscription(Int32, '/inference32', self.listener_callback, qos_profile)
        self.subscription  # prevent unused variable warning
        self.bool_subscription = self.create_subscription(Bool, '/bool', self.bool_callback, qos_profile)
        # subscriber for update score at each correct prediction
        self.score_sub = self.create_subscription(Bool, '/score', self.score_callback, qos_profile)
        # for publishing each time I press a buttom that changes the state
        self.mode_pub = self.create_publisher(Int32, '/mode32', qos_profile=rclpy.qos.qos_profile_sensor_data)
        # for publishing each time I change the label in recording mode
        self.label_pub = self.create_publisher(Int32, '/label32', qos_profile=rclpy.qos.qos_profile_sensor_data)
        # Publisher for participnat info in recording mode
        self.participant_info_pub = self.create_publisher(RecordingFolder, '/folder_info', qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.participant_info_pub_emg = self.create_publisher(RecordingFolder, '/folder_info_emg', qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.participant_info_pub_raw_imu = self.create_publisher(RecordingFolder, '/folder_info_raw_imu', qos_profile=rclpy.qos.qos_profile_sensor_data)
        # Training path publisher
        self.training_path_pub = self.create_publisher(String, '/training_path', qos_profile=rclpy.qos.qos_profile_sensor_data)
        print("Initializing GUI...")  # Debug print
        self.window = tk.Tk()
        self.window.title("ROS2 Multi-Object GUI")

        self.canvas = tk.Canvas(self.window, width=1300, height=900)
        self.canvas.pack()

        self.inference_count = 0

        # Load and set background image
        try:
            self.bg_image = Image.open("CBPR_images/shelf.jpg")  # Replace with your background image
            self.bg_image = self.bg_image.resize((1300, 900), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_photo)
        except Exception as e:
            print(f"Error loading background image: {str(e)}")

        # Define image dimensions
        self.image_width = 200
        self.image_height = 200

        # Load and create image objects
        self.objects = []
        object_positions = [(330, 290),  (940, 290), (330, 650), (940, 650)]
        
        for i, pos in enumerate(object_positions):
            try:
                print(f"Loading image for object {i}...")  # Debug print
                original_image = Image.open(f"CBPR_images/obj_{i+1}.png")  # Replace with your images
                original_image = original_image.resize((self.image_width, self.image_height), Image.LANCZOS)

                gray_image = original_image.copy()
                gray_image = ImageEnhance.Color(gray_image).enhance(0.0)
                
                tk_gray_image = ImageTk.PhotoImage(gray_image)
                tk_color_image = ImageTk.PhotoImage(original_image)
                
                image_item = self.canvas.create_image(pos[0], pos[1], image=tk_gray_image)
                
                self.objects.append({
                    'id': i,
                    'item': image_item,
                    'gray': tk_gray_image,
                    'color': tk_color_image,
                    'state': 'gray'
                })
                print(f"Object {i} loaded successfully.")  # Debug print
            except Exception as e:
                print(f"Error loading image for object {i}: {str(e)}")  # Debug print

        # Create game-style font
        game_font = tkfont.Font(family="Arial", size=24, weight="bold")
        # Create text object with game-style font
        self.text_item = self.canvas.create_text(650, 460, text="GAME TEXT", font=game_font, fill="red")
        
        # BUTTONS
        # Create a frame for buttons
        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(pady=10)

        # Add Work/Not Work toggle button
        self.work_button = ToggleButton(self.button_frame, width=100, height=30, label="WORKING", command=self.toggle_work_mode)
        self.work_button.grid(row=0, column=0, padx=5)

        # Add two additional toggle buttons
        self.rec_button = ToggleButton(self.button_frame, width=100, height=30, label="RECORDING", command=self.rec_button_action)
        self.rec_button.grid(row=0, column=1, padx=5)
        self.rec_button.set_enabled(False)

        self.train_button = ToggleButton(self.button_frame, width=100, height=30, label="TRAINING", command=self.train_button_action)
        self.train_button.grid(row=0, column=2, padx=5)
        self.train_button.set_enabled(False)

        # Add a new frame for recording buttons
        self.recording_frame = tk.Frame(self.window)
        self.recording_frame.pack(pady=10)
        self.recording_buttons = []

        self.last_update_time = self.get_clock().now()
        self.update_interval = 2.0  # seconds

        # Initialize score
        self.score = 0
        # Create and place the score label
        self.score_label = tk.Label(self.window, text=f"SCORE: {self.score}", bg="#f0f0f0", font=("Arial", 24, "bold"))
        self.score_label.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)
        
        print("GUI initialization complete.")  # Debug print

        # Initialize participant info variable
        self.participant_info = None

        global BASE_DIR
        self.root = tk.Toplevel(self.window)
        dialog = ParticipantInfoDialogAfterInit(self.root, "Participant Information")
        self.root.destroy()
        participant_id = dialog.participant_id
        repository = dialog.repository

        # Modify the BASE_DIR path
        BASE_DIR = os.path.join(BASE_DIR, repository)

        # Save participant information to a text file
        self.save_participant_info(dialog)

        # Add new attributes for recording mode
        self.recording_mode = False
        self.current_object = None
        self.rectangle = None
        self.rectangle_color = "green"

        # New attributes for rectangle configuration
        self.rectangle_size = (150, 150)  # (width, height)
        self.rectangle_position = "top-left"  # Can be "top-left", "top-right", "bottom-left", "bottom-right"
        self.color_change_count = 0
        self.position_change_threshold = 4  # Number of color changes before position change
        self.positions = ["top-left", "top-right", "bottom-right", "bottom-left"]
        self.model_info_list = []

        # List to store the sequence of button numbers
        self.rest = True
        self.current_sequence_button = 1

        # Bind the Enter key to the handle_enter method
        self.window.bind('<Return>', self.handle_enter)

        self.counter_training_finished = 0
        self.previous_button = 0

    def handle_enter(self, event=None):
        # Get the button widget
        if self.rest:
            button = self.current_sequence_button
            self.current_sequence_button += 1
            if self.current_sequence_button == 5:
                self.current_sequence_button = 1
        else:
            button = 5
        # Trigger the button command (simulates button press logic)
        self.recording_button_action(button)
        self.rest = not self.rest
        print(f"\n\n\n\n\n\n\n\n Button pressed: {button} \n\n\n\n\n\n\n\n")  # Debug print

    def create_initial_rectangle(self):
        if self.rectangle is None:
            self.update_rectangle()

    def update_rectangle(self):
        rect_coords = self.get_rectangle_coordinates()
        if self.rectangle is None:
            self.rectangle = self.canvas.create_rectangle(*rect_coords, fill=self.rectangle_color, outline="")
        else:
            self.canvas.coords(self.rectangle, *rect_coords)
        self.canvas.itemconfig(self.rectangle, fill=self.rectangle_color)

    def get_rectangle_coordinates(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        rect_width, rect_height = self.rectangle_size

        padding = 200  # Padding from the canvas edges
        center_offset = 100  # Amount to move rectangles towards the center

        if self.rectangle_position == "top-left":
            return (padding + center_offset, padding, 
                    padding + center_offset + rect_width, padding + rect_height)
        elif self.rectangle_position == "top-right":
            return (canvas_width - padding - rect_width - center_offset, padding, 
                    canvas_width - padding - center_offset, padding + rect_height)
        elif self.rectangle_position == "bottom-left":
            return (padding + center_offset, canvas_height - padding - rect_height, 
                    padding + center_offset + rect_width, canvas_height - padding)
        elif self.rectangle_position == "bottom-right":
            return (canvas_width - padding - rect_width - center_offset, canvas_height - padding - rect_height, 
                    canvas_width - padding - center_offset, canvas_height - padding)
        else:
            return (padding + center_offset, padding, 
                    padding + center_offset + rect_width, padding + rect_height) # Default to top-left

    def set_rectangle_size(self, width, height):
        self.rectangle_size = (width, height)
        if self.rectangle:
            self.update_rectangle()

    def set_rectangle_position(self, position):
        valid_positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
        if position in valid_positions:
            self.rectangle_position = position
            if self.rectangle:
                self.update_rectangle()
        else:
            self.get_logger().warn(f'Invalid rectangle position: {position}')

    def save_participant_info(self, dialog):
        global BASE_DIR
        info_file = os.path.join(BASE_DIR, "PARTICIPANT_INFO.txt")
        os.makedirs(BASE_DIR, exist_ok=True)
        with open(info_file, "w") as f:
            f.write(f"Participant ID: {dialog.participant_id}\n")
            f.write(f"Repository: {dialog.repository}\n")
            f.write(f"Name: {dialog.name}\n")
            f.write(f"Gender: {dialog.gender}\n")
            f.write(f"Age: {dialog.age}\n")
            f.write(f"Height (cm): {dialog.height}\n")
            f.write(f"Weight (kg): {dialog.weight}\n")
            f.write(f"Upper Arm Length (cm): {dialog.upper_arm_length}\n")
            f.write(f"Forearm Length (cm): {dialog.forearm_length}\n")

    def toggle_work_mode(self, state):
        if state:
            self.rec_button.set_enabled(True)
            self.train_button.set_enabled(True)
        else:
            self.rec_button.set_enabled(False)
            self.train_button.set_enabled(False)
            if self.rec_button.switch_on:
                self.rec_button.set_state(False)
                self.rec_button_action(False)
            if self.train_button.switch_on:
                self.train_button.set_state(False)
                self.train_button_action(False)
            self.hide_recording_buttons()  # Hide recording buttons when not working
        self.publish_state(1 if state else 0)
        print(f"Work mode: {state}")  # Debug print

    def rec_button_action(self, state):
        global number_recordings
        if self.work_button.switch_on:
            if state:
                self.recording_mode = True
                self.window.after(100, self.create_initial_rectangle)
                # Publish participant info
                msg = RecordingFolder()
                msg.folder_path = f"{BASE_DIR}/DATASET_ANGLE_ESTIMATION_{number_recordings-1}"
                self.participant_info_pub.publish(msg)
                msg.folder_path = f"{BASE_DIR}/DATASET_EMG_{number_recordings-1}"
                self.participant_info_pub_emg.publish(msg)
                msg.folder_path = f"{BASE_DIR}/DATASET_RAW_IMU_{number_recordings-1}"
                self.participant_info_pub_raw_imu.publish(msg)
                self.train_button.set_state(False)
                self.show_recording_buttons()  # Show buttons when recording
                self.publish_state(3 if state else 1)
                number_recordings += 1  
            else:
                self.recording_mode = False
                self.hide_recording_buttons()
                self.reset_gui_state()
                self.hide_recording_buttons()  # Hide buttons when not recording
                self.publish_state(4)
    
    def collect_participant_info(self):
        if len(self.model_info_list) == 3:
            param1, param2, param3 = self.model_info_list
            self.model_info_list = []
            dialog = ParticipantInfoDialog(self.window, param1, param2, param3)
            self.window.wait_window(dialog)
        return dialog

    def train_button_action(self, state):
        global training_path
        global BASE_DIR
        if self.work_button.switch_on:
            if state:
                folder_path = TrainingPanel(self.window, BASE_DIR)
                self.window.wait_window(folder_path)
                print()
                print(training_path)
                msg = String()
                msg.data = str(BASE_DIR+'/'+training_path)
                self.training_path_pub.publish(msg)
                self.rec_button.set_state(False)
                self.hide_recording_buttons()  # Hide buttons when training
            self.publish_state(2 if state else 1)

    def show_recording_buttons(self):
        self.recording_frame.pack(pady=10)
        for i in range(4):
            btn = tk.Button(self.recording_frame, text=f"Button {i+1}", 
                            command=lambda i=i: self.recording_button_action(i+1))
            btn.grid(row=0, column=i, padx=5, pady=5)
            self.recording_buttons.append(btn)
        self.rectangle_color = "red"
        self.update_rectangle()
        self.show_object_in_center(0)

    def hide_recording_buttons(self):
        for btn in self.recording_buttons:
            btn.destroy()
        self.recording_buttons.clear()
        self.recording_frame.pack_forget()

    def recording_button_action(self, button_number):
        print(f"Button {button_number} pressed.")
        msg = Int32()
        msg.data = button_number
        self.label_pub.publish(msg)
        self.get_logger().info(f"Recording button {button_number} pressed and sent.")

        if self.recording_mode:
            if 1 <= button_number <= 4:
                self.show_object_in_center(button_number - 1)
                self.rectangle_color = "green"
                self.update_rectangle()
            elif button_number == 5:
                self.toggle_rectangle_color_to_rest()
                if self.previous_button == 4:
                    self.previous_button = 0
                self.show_object_in_center(self.previous_button)
            if button_number == 5:
                self.previous_button = 1
            else:
                self.previous_button = button_number # previous button is actually the next object to show

    def show_object_in_center(self, object_id):
        # Hide all objects
        for obj in self.objects:
            self.canvas.itemconfig(obj['item'], state='hidden')

        # Show the selected object in the center
        target_obj = self.objects[object_id]
        self.canvas.itemconfig(target_obj['item'], state='normal')
        self.canvas.itemconfig(target_obj['item'], image=target_obj['gray'])
        self.canvas.coords(target_obj['item'], 650, 450)  # Center of the canvas
        self.current_object = object_id

        # Update the rectangle (don't create a new one)
        self.update_rectangle()

    def toggle_rectangle_color_to_rest(self):
        self.rectangle_color = "red"# if self.rectangle_color == "green" else "green"
        self.color_change_count += 1
        
        if self.color_change_count >= self.position_change_threshold:
            self.change_rectangle_position()
            self.color_change_count = 0

        if self.rectangle:
            self.canvas.itemconfig(self.rectangle, fill=self.rectangle_color)            

    def change_rectangle_position(self):
        positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
        current_index = positions.index(self.rectangle_position)
        next_index = (current_index + 1) % len(positions)
        self.set_rectangle_position(positions[next_index])

    def publish_state(self, state):
        msg = Int32()
        msg.data = state
        self.mode_pub.publish(msg)
        if state == 4:
            state = 1
        self.text_update(state)

    def listener_callback(self, msg):
        self.inference_count += 1
        print(f"Inference count: {self.inference_count}")
        self.last_update_time = self.get_clock().now()

        try:
            object_id = int(msg.data)
            # TO DELETE WITH OTHER DATA
            if object_id == 5 or object_id == 6 or object_id == 4:
                object_id -= 3
            if 0 <= object_id < len(self.objects):
                if self.recording_mode and self.current_object is not None:
                    if object_id == self.current_object:
                        self.canvas.itemconfig(self.objects[object_id]['item'], image=self.objects[object_id]['color'])
                    else:
                        self.canvas.itemconfig(self.objects[self.current_object]['item'], image=self.objects[self.current_object]['gray'])
                else:
                    # Original behavior for non-recording mode
                    for obj in self.objects:
                        if obj['state'] == 'color':
                            self.canvas.itemconfig(obj['item'], image=obj['gray'])
                            obj['state'] = 'gray'
                    
                    target_obj = self.objects[object_id]
                    self.canvas.itemconfig(target_obj['item'], image=target_obj['color'])
                    target_obj['state'] = 'color'
            else:
                self.get_logger().warn(f'Invalid object ID: {object_id}')
        except ValueError:
            self.get_logger().warn(f'Invalid message format: {msg.data}')

    def score_callback(self, msg):
        self.score += 1
        self.score_label.config(text=f"SCORE: {self.score}")

    def bool_callback(self, msg):
        self.counter_training_finished += 1
        if self.counter_training_finished == 3:
            self.work_button.set_state(False)
            self.toggle_work_mode(False)
            self.train_button.set_state(False)
            self.train_button_action(False)
            # Publish the state changes
            self.publish_state(0)  # 0 for not working
            self.counter_training_finished = 0
        else:
            # switch to simple working mode
            self.toggle_work_mode(True)
            self.work_button.set_state(True)
            self.train_button.set_state(False)
            self.train_button_action(False)
            self.publish_state(1)  # 1 for working


    def text_update(self, index):
        self.canvas.itemconfig(self.text_item, text=MSGS_LIST[index])

    def check_update(self):
        current_time = self.get_clock().now()
        if (current_time - self.last_update_time).nanoseconds / 1e9 > self.update_interval:
            for obj in self.objects:
                if obj['state'] == 'color':
                    self.canvas.itemconfig(obj['item'], image=obj['gray'])
                    obj['state'] = 'gray'
            #print("Reset all objects to gray state.")  # Debug print

        self.window.after(100, self.check_update)  # Check every 100ms

    def reset_gui_state(self):
        # Reset the GUI state when exiting recording mode
        for obj in self.objects:
            self.canvas.itemconfig(obj['item'], state='normal')
            self.canvas.itemconfig(obj['item'], image=obj['gray'])
        
        if self.rectangle:
            self.canvas.delete(self.rectangle)
            self.rectangle = None
        
        self.current_object = None
        self.rectangle_color = "green"

def main(args=None):
    print("Starting ROS2 node...")  # Debug print
    rclpy.init(args=args)
    gui = MultiObjectGUI()

    print("Starting GUI update loop...")  # Debug print
    gui.check_update()  # Start the periodic check

    print("Entering ROS2 spin...")  # Debug print
    while rclpy.ok():
        rclpy.spin_once(gui, timeout_sec=0.1)
        gui.window.update()

    print("ROS2 shutting down...")  # Debug print
    gui.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
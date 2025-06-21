import cv2
import mediapipe as mp
import numpy as np
import time
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import itertools
import copy
import os
import logging
import sys
import json
import pickle
from pathlib import Path
from sklearn.neural_network import MLPClassifier

# Suppress TensorFlow Lite and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

# Redirect stderr for Abseil logs
class SuppressStderr:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.stderr

# Initialize MediaPipe Hands
with SuppressStderr():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

# App configuration
APP_NAME = "Sign Language Creator"
BASE_DIR = Path.home() / APP_NAME
PROJECTS_DIR = BASE_DIR / "data" / "projects"
DEFAULT_PROJECT = "default"

# Ensure base directories exist
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage
gestures = {}
training_data = {'X': [], 'y': []}
classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
current_project = DEFAULT_PROJECT

# File paths for current project
def get_project_paths(project_name):
    project_dir = PROJECTS_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    return {
        'gestures': project_dir / "gestures.json",
        'training_data': project_dir / "training_data.json",
        'model': project_dir / "model.pkl"
    }

# Load data from project
def load_data(project_name, status_label):
    global gestures, training_data, classifier
    paths = get_project_paths(project_name)
    try:
        gestures.clear()
        training_data['X'].clear()
        training_data['y'].clear()
        classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        
        if paths['gestures'].exists():
            with open(paths['gestures'], 'r') as f:
                gestures.update(json.load(f))
        
        if paths['training_data'].exists():
            with open(paths['training_data'], 'r') as f:
                loaded_data = json.load(f)
                training_data['X'] = [np.array(x) for x in loaded_data['X']]
                training_data['y'] = loaded_data['y']
        
        if paths['model'].exists():
            with open(paths['model'], 'rb') as f:
                classifier = pickle.load(f)
        
        status_label.configure(text=f"Loaded project: {project_name}", text_color="green")
        return True
    except Exception as e:
        status_label.configure(text=f"Error loading project: {str(e)}", text_color="red")
        return False

# Save data to project
def save_data(status_label):
    paths = get_project_paths(current_project)
    try:
        with open(paths['gestures'], 'w') as f:
            json.dump(gestures, f, indent=4)
        
        with open(paths['training_data'], 'w') as f:
            json.dump({
                'X': [x.tolist() for x in training_data['X']],
                'y': training_data['y']
            }, f, indent=4)
        
        with open(paths['model'], 'wb') as f:
            pickle.dump(classifier, f)
        
        status_label.configure(text="Data saved successfully.", text_color="green")
        return True
    except Exception as e:
        status_label.configure(text=f"Error saving data: {str(e)}", text_color="red")
        return False

# Calculate bounding square
def calc_bounding_square(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    size = max(w, h) + 20
    x_center = x + w // 2
    y_center = y + h // 2
    x_min = max(0, x_center - size // 2)
    x_max = min(image_width, x_center + size // 2)
    y_min = max(0, y_center - size // 2)
    y_max = min(image_height, y_center + size // 2)

    return [x_min, y_min, x_max, y_max]

# Pre-process landmarks
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for landmark in temp_landmark_list:
        landmark[0] -= base_x
        landmark[1] -= base_y
    max_value = max([abs(coord) for landmark in temp_landmark_list for coord in landmark] or [1])
    for landmark in temp_landmark_list:
        landmark[0] /= max_value
        landmark[1] /= max_value
    return list(itertools.chain.from_iterable(temp_landmark_list))

# Capture landmarks
def capture_landmarks(cap, duration=5, status_label=None):
    landmarks_list = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.resize(image, (640, 480))
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        results.image_width = 640  # Set image dimensions to suppress NORM_RECT warning
        results.image_height = 480
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_points = [[lm.x * 640, lm.y * 480] for lm in hand_landmarks.landmark]
                landmarks_list.append(pre_process_landmark(landmark_points))
        
        cv2.imshow("Recording Gesture", image)
        if status_label:
            status_label.configure(text=f"Recording gesture... {duration - (time.time() - start_time):.1f}s remaining")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow("Recording Gesture")
    return landmarks_list

# Train classifier
def train_classifier(status_label):
    if not training_data['X'] or not training_data['y']:
        status_label.configure(text="Error: No training data available.", text_color="red")
        return False
    status_label.configure(text="Training model...", text_color="yellow")
    try:
        classifier.fit(training_data['X'], training_data['y'])
        save_data(status_label)
        status_label.configure(text="Model trained successfully.", text_color="green")
        return True
    except Exception as e:
        status_label.configure(text=f"Training error: {str(e)}", text_color="red")
        return False

# Recognize gestures
def recognize_gestures(gestures, status_label):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_label.configure(text="Error: Could not open webcam.", text_color="red")
        return
    
    last_detected = 0
    current_subtitle = ""
    subtitle_start_time = 0
    subtitle_duration = 3.5
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.resize(image, (640, 480))
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            results.image_width = 640  # Set image dimensions to suppress NORM_RECT warning
            results.image_height = 480
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    brect = calc_bounding_square(image, hand_landmarks)
                    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (255, 255, 255), 2)
                    
                    landmark_points = [[lm.x * 640, lm.y * 480] for lm in hand_landmarks.landmark]
                    current_landmarks = pre_process_landmark(landmark_points)
                    
                    if training_data['X'] and time.time() - last_detected > 2:
                        try:
                            prediction = classifier.predict([current_landmarks])[0]
                            if prediction in gestures:
                                current_subtitle = gestures[prediction].get('subtitle', '')
                                subtitle_start_time = time.time()
                                status_label.configure(text=f"Recognized: {prediction}")
                                last_detected = time.time()
                        except ValueError as e:
                            status_label.configure(text=f"Prediction error: {str(e)}", text_color="red")
                    
                    if time.time() - subtitle_start_time < subtitle_duration and current_subtitle:
                        text_size = cv2.getTextSize(current_subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        text_width = text_size[0]
                        text_x = max(0, min(brect[0] + (brect[2] - brect[0] - text_width) // 2, 640 - text_width))
                        text_y = max(30, brect[1] - 15)
                        
                        overlay = image.copy()
                        rect_x_min = text_x - 10
                        rect_x_max = text_x + text_width + 10
                        rect_y_min = text_y - text_size[1] - 10
                        rect_y_max = text_y + 10
                        cv2.rectangle(overlay, (rect_x_min, rect_y_min), (rect_x_max, rect_y_max), (0, 0, 0), -1)
                        alpha = 0.6
                        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                        
                        cv2.putText(
                            image,
                            current_subtitle,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
            
            cv2.imshow("Sign Language Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# GUI Application
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root.geometry("600x600")
        self.root.resizable(True, True)
        
        # Try to set app icon
        try:
            icon_path = Path(__file__).parent / "icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        # Menu bar
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Project", command=self.new_project)
        filemenu.add_command(label="Open Project", command=self.open_project)
        filemenu.add_command(label="Save Project", command=self.save_project)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menubar)
        
        # Main frame
        self.frame = ctk.CTkFrame(root, corner_radius=10)
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Input fields
        ctk.CTkLabel(self.frame, text="Sign Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.name_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="Enter sign name")
        self.name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ctk.CTkLabel(self.frame, text="Subtitle:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.subtitle_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="Enter subtitle text")
        self.subtitle_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Buttons
        ctk.CTkButton(self.frame, text="1. Record Gesture", command=self.record_gesture).grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        ctk.CTkButton(self.frame, text="2. Add Subtitle", command=self.add_subtitle).grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        ctk.CTkButton(self.frame, text="3. Start Recognition", command=self.start_recognition).grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")
        
        # Gesture list
        ctk.CTkLabel(self.frame, text="Saved Gestures:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.gesture_listbox = tk.Listbox(self.frame, height=10, width=50)
        self.gesture_listbox.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Delete and Clear buttons
        ctk.CTkButton(self.frame, text="Delete Selected Gesture", command=self.delete_gesture).grid(row=7, column=0, pady=5, sticky="ew")
        ctk.CTkButton(self.frame, text="Clear All Data", command=self.clear_data).grid(row=7, column=1, pady=5, sticky="ew")
        
        # Status bar
        self.status_label = ctk.CTkLabel(self.frame, text=f"Project: {current_project}", text_color="gray", anchor="w")
        self.status_label.grid(row=8, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Configure grid weights
        self.frame.columnconfigure((0, 1), weight=1)
        self.frame.rowconfigure(6, weight=1)
        
        # Load default project
        self.load_project(DEFAULT_PROJECT)
        self.update_gesture_list()
    
    def update_gesture_list(self):
        self.gesture_listbox.delete(0, tk.END)
        for name in gestures:
            self.gesture_listbox.insert(tk.END, f"{name}: {gestures[name]['subtitle']}")
    
    def new_project(self):
        project_name = tk.simpledialog.askstring("New Project", "Enter project name:", parent=self.root)
        if project_name and project_name.strip():
            global current_project
            current_project = project_name.strip()
            self.load_project(current_project)
            self.status_label.configure(text=f"Created project: {current_project}", text_color="green")
    
    def open_project(self):
        project_name = tk.simpledialog.askstring("Open Project", "Enter project name:", parent=self.root)
        if project_name and project_name.strip():
            if (PROJECTS_DIR / project_name).exists():
                global current_project
                current_project = project_name.strip()
                self.load_project(current_project)
                self.status_label.configure(text=f"Opened project: {current_project}", text_color="green")
            else:
                self.status_label.configure(text="Project not found.", text_color="red")
    
    def save_project(self):
        if save_data(self.status_label):
            self.status_label.configure(text=f"Saved project: {current_project}", text_color="green")
    
    def load_project(self, project_name):
        if load_data(project_name, self.status_label):
            self.update_gesture_list()
    
    def show_about(self):
        messagebox.showinfo("About", f"{APP_NAME}\nVersion 1.0\nA desktop app for creating and recognizing custom sign language gestures.\n© 2025")
    
    def record_gesture(self):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.configure(text="Error: Enter a sign name.", text_color="red")
            return
        
        if name in gestures:
            self.status_label.configure(text="Error: Sign name already exists.", text_color="red")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_label.configure(text="Error: Could not open webcam.", text_color="red")
            return
        
        try:
            self.status_label.configure(text="Recording gesture for 5 seconds...", text_color="yellow")
            landmarks = capture_landmarks(cap, duration=5, status_label=self.status_label)
            
            if landmarks:
                gestures[name] = {'landmarks': landmarks, 'subtitle': ''}  # Store landmarks directly
                for lm in landmarks:
                    training_data['X'].append(np.array(lm))  # Convert to NumPy array for training
                    training_data['y'].append(name)
                save_data(self.status_label)
                if train_classifier(self.status_label):
                    self.status_label.configure(text=f"Gesture '{name}' recorded and model trained.", text_color="green")
                else:
                    self.status_label.configure(text="Gesture recorded but training failed.", text_color="red")
                self.update_gesture_list()
            else:
                self.status_label.configure(text="Error: No hand detected.", text_color="red")
        finally:
            cap.release()
    
    def add_subtitle(self):
        name = self.name_entry.get().strip()
        subtitle = self.subtitle_entry.get().strip()
        if not name:
            self.status_label.configure(text="Error: Enter a sign name.", text_color="red")
            return
        
        if name not in gestures:
            self.status_label.configure(text="Error: Record gesture first.", text_color="red")
            return
        
        if not subtitle:
            self.status_label.configure(text="Error: Enter a subtitle.", text_color="red")
            return
        
        gestures[name]['subtitle'] = subtitle
        save_data(self.status_label)
        self.update_gesture_list()
        self.status_label.configure(text=f"Subtitle for '{name}' added.", text_color="green")
    
    def delete_gesture(self):
        selection = self.gesture_listbox.curselection()
        if not selection:
            self.status_label.configure(text="Error: Select a gesture to delete.", text_color="red")
            return
        
        name = self.gesture_listbox.get(selection[0]).split(":")[0]
        if name in gestures:
            del gestures[name]
            indices = [i for i, label in enumerate(training_data['y']) if label == name]
            training_data['X'] = [x for i, x in enumerate(training_data['X']) if i not in indices]
            training_data['y'] = [y for i, y in enumerate(training_data['y']) if i not in indices]
            save_data(self.status_label)
            if training_data['X']:
                train_classifier(self.status_label)
            self.update_gesture_list()
            self.status_label.configure(text=f"Gesture '{name}' deleted.", text_color="green")
    
    def clear_data(self):
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all gestures and reset the model?"):
            global gestures, training_data, classifier
            gestures.clear()
            training_data['X'].clear()
            training_data['y'].clear()
            classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            save_data(self.status_label)
            self.update_gesture_list()
            self.status_label.configure(text="All data cleared.", text_color="green")
    
    def start_recognition(self):
        if not gestures or not training_data['X']:
            self.status_label.configure(text="Error: No gestures recorded.", text_color="red")
            return
        self.status_label.configure(text="Starting recognition...", text_color="yellow")
        threading.Thread(target=recognize_gestures, args=(gestures, self.status_label), daemon=True).start()

# Main execution
if __name__ == "__main__":
    root = ctk.CTk()
    app = SignLanguageApp(root)
    root.mainloop()

# Cleanup
hands.close()
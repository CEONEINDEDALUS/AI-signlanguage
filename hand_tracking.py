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
from contextlib import redirect_stderr
from sklearn.neural_network import MLPClassifier

# Suppress TensorFlow Lite and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = ERROR only
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)  # Stricter filtering
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

# Redirect stderr for Abseil logs
class SuppressStderr:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.stderr

# Initialize MediaPipe Hands with explicit image dimensions
with SuppressStderr():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

# In-memory storage for gestures and training data
gestures = {}
training_data = {'X': [], 'y': []}  # X: landmark features, y: gesture labels
classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Calculate bounding square around hand
def calc_bounding_square(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    size = max(w, h) + 20  # Add padding
    x_center = x + w // 2
    y_center = y + h // 2
    x_min = max(0, x_center - size // 2)
    x_max = min(image_width, x_center + size // 2)
    y_min = max(0, y_center - size // 2)
    y_max = min(image_height, y_center + size // 2)

    return [x_min, y_min, x_max, y_max]

# Pre-process landmarks for ML model
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for landmark in temp_landmark_list:
        landmark[0] -= base_x
        landmark[1] -= base_y
    
    # Normalize
    max_value = max([abs(coord) for landmark in temp_landmark_list for coord in landmark] or [1])
    for landmark in temp_landmark_list:
        landmark[0] /= max_value
        landmark[1] /= max_value
    
    # Flatten to one-dimensional list (21 landmarks × 2 coordinates = 42 features)
    return list(itertools.chain.from_iterable(temp_landmark_list))

# Capture hand landmarks and store for training
def capture_landmarks(cap, duration=5, status_label=None):
    landmarks_list = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        success, image = cap.read()
        if not success:
            continue
        
        # Resize image to fixed dimensions
        image = cv2.resize(image, (640, 480))
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Get pixel coordinates
                landmark_points = [[lm.x * 640, lm.y * 480] for lm in hand_landmarks.landmark]
                landmarks_list.append(pre_process_landmark(landmark_points))
        
        cv2.imshow("Recording Gesture", image)
        if status_label:
            status_label.configure(text="Recording gesture... Keep gesturing for 5 seconds.")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow("Recording Gesture")
    return landmarks_list

# Train or retrain the ML model
def train_classifier(status_label):
    if not training_data['X'] or not training_data['y']:
        return False
    status_label.configure(text="Training model...", text_color="yellow")
    try:
        classifier.fit(training_data['X'], training_data['y'])
        status_label.configure(text="Model trained successfully.", text_color="green")
        return True
    except Exception as e:
        status_label.configure(text=f"Training error: {str(e)}", text_color="red")
        return False

# Real-time gesture recognition with ML model
def recognize_gestures(gestures, status_label):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_label.configure(text="Error: Could not open webcam.", text_color="red")
        return
    
    last_detected = 0
    current_subtitle = ""
    subtitle_start_time = 0
    subtitle_duration = 3.5
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        # Resize image to fixed dimensions
        image = cv2.resize(image, (640, 480))
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw white square around hand
                brect = calc_bounding_square(image, hand_landmarks)
                cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (255, 255, 255), 2)
                
                # Get landmarks for prediction
                landmark_points = [[lm.x * 640, lm.y * 480] for lm in hand_landmarks.landmark]
                current_landmarks = pre_process_landmark(landmark_points)
                
                # Predict gesture using ML model
                if training_data['X'] and time.time() - last_detected > 2:
                    try:
                        prediction = classifier.predict([current_landmarks])[0]
                        if prediction in gestures:
                            current_subtitle = gestures[prediction].get('subtitle', '')
                            subtitle_start_time = time.time()
                            status_label.configure(text=f"Recognized: {prediction}")
                            last_detected = time.time()
                    except:
                        pass
                
                # Display subtitle above square
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
    
    cap.release()
    cv2.destroyAllWindows()

# GUI Application
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Easy Sign Language Creator")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # GUI Elements
        self.frame = ctk.CTkFrame(root, corner_radius=10)
        self.frame.grid(padx=20, pady=20, sticky="nsew")
        
        # Sign Name
        ctk.CTkLabel(self.frame, text="Sign Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.name_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="Enter sign name")
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Subtitle
        ctk.CTkLabel(self.frame, text="Subtitle:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.subtitle_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="Enter subtitle text")
        self.subtitle_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Status Label
        self.status_label = ctk.CTkLabel(self.frame, text="Ready", text_color="gray")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Buttons
        ctk.CTkButton(self.frame, text="1. Record Gesture", command=self.record_gesture).grid(row=3, column=0, columnspan=2, pady=5)
        ctk.CTkButton(self.frame, text="2. Add Subtitle", command=self.add_subtitle).grid(row=4, column=0, columnspan=2, pady=5)
        ctk.CTkButton(self.frame, text="3. Start Recognition", command=self.start_recognition).grid(row=5, column=0, columnspan=2, pady=5)
    
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
        
        self.status_label.configure(text="Recording gesture for 5 seconds...", text_color="yellow")
        landmarks = capture_landmarks(cap, duration=5, status_label=self.status_label)
        cap.release()
        
        if landmarks:
            gestures[name] = {'landmarks': landmarks, 'subtitle': ''}
            # Add to training data
            for lm in landmarks:
                training_data['X'].append(lm)
                training_data['y'].append(name)
            # Train model
            if train_classifier(self.status_label):
                self.status_label.configure(text=f"Gesture '{name}' recorded and model trained. Now add subtitle.", text_color="green")
            else:
                self.status_label.configure(text="Gesture recorded but training failed.", text_color="red")
        else:
            self.status_label.configure(text="Error: No hand detected.", text_color="red")
    
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
        self.status_label.configure(text=f"Subtitle for '{name}' added.", text_color="green")
    
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
    root.geometry("400x300")
    root.mainloop()

# Cleanup
hands.close()
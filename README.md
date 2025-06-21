# Sign Language Creator


https://github.com/user-attachments/assets/d461d132-fd47-4507-8c54-cc326034abc6


**Sign Language Creator** is a desktop application for creating, training, and recognizing custom sign language gestures using your webcam. It allows users to define their own signs, add subtitles, train a model, and perform real-time sign recognition.

## Features

- **Create Custom Signs:** Record new hand gestures and save them with unique names.
- **Add Subtitles:** Assign subtitles or text to each gesture.
- **Project Management:** Save, open, and manage multiple projects.
- **Real-Time Recognition:** Recognize trained gestures live using your webcam, with subtitles displayed as overlays.
- **User-Friendly GUI:** Built with CustomTkinter for a modern look.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CEONEINDEDALUS/AI-signlanguage.git
   cd AI-signlanguage
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python customsignlangugae.py
   ```

## Requirements

- Python 3.8 or newer
- See [`requirements.txt`](requirements.txt) for Python dependencies

## Usage

1. **Record Gestures:**  
   Enter a sign name and click "Record Gesture" to capture a gesture using your webcam.

2. **Add Subtitle:**  
   Enter a subtitle and click "Add Subtitle" to label the gesture.

3. **Start Recognition:**  
   Click "Start Recognition" to recognize signs in real-time.

4. **Manage Gestures:**  
   - Use the list to view saved gestures.
   - Delete or clear gestures as needed.
   - Create or switch between projects via the File menu.

## Folders and Files

- `customsignlangugae.py` — Main application file.
- `data/projects/` — Where project data is stored (created on first run).
- `icon.ico` (optional) — Application icon.

## Notes

- **Webcam Required** for gesture recording and recognition.
- **Windows, Mac, Linux** supported (some dependencies may require additional system packages).
- **All data is stored locally** in your home folder.

## Troubleshooting

- If the webcam is not detected, ensure no other app is using it.
- For issues with missing dependencies, run:
  ```
  pip install -r requirements.txt
  ```
- If the GUI fails to open, check your Python and Tkinter installation.

## License

© 2025 CEONEINDEDALUS  
For personal, educational, and non-commercial use.

# ✋ Real-Time Hand Gesture Recognition

This project implements **real-time static hand gesture recognition** using **MediaPipe, OpenCV, and TensorFlow Lite**.  
Demo: https://drive.google.com/file/d/1BZ6viH2rCFhZeYTYkAvUNLq7iehFB6VT/view?usp=drivesdk

---

## Features
- Detects a single hand in real time using **MediaPipe Hands**.
- Classifies gestures using a trained **TensorFlow Lite model**.
- Supports **logging mode** to collect custom training data (`keypoint.csv`).
- Custom gestures can be added by collecting data and retraining the model.
- Toggle between **inference mode** and **logging mode** with a single key.
- Lightweight and runs in real-time even on a laptop GPU/CPU.

---

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd HandGestureRecognition
```

### 2. Create virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # (Windows)
source .venv/bin/activate    # (Linux/Mac)
```

### 3. Install dependencies
For running the app:
```bash
pip install -r requirements.txt
```

For training (Jupyter Notebook):
```bash
pip install -r requirements_training.txt
```

---

## Running the Application
```bash
python app.py
```

### Controls:
- **Default mode = Inference** (if trained model exists).
- **Press `k`** → toggle Logging mode ON/OFF.
- **Press `0–9` keys** → assign the shown gesture to that number label (saved in `keypoint.csv`).
- **Press `ESC`** → exit program.
- **Or `Ctrl + C`** if running from terminal.

---

## Training New Gestures
1. Run `app.py`, press `k` to enter logging mode.  
2. Show a gesture and press a digit key (0–9) to log it.  
   - Collect ~200–300 samples per gesture.  
3. After collecting enough data → open the notebook:
   ```bash
   jupyter notebook training/keypoint_classification_EN.ipynb
   ```
4. Run all cells → this will:
   - Train a neural network.
   - Export `keypoint_classifier.tflite`.
   - Export `keypoint_classifier_label.csv`.
5. Restart `app.py` → now the app will classify your gestures in real time 🎉

---

## File Overview
- **app.py** → Main app (inference + logging).  
- **utils.py** → FPS calculator.  
- **model/keypoint_classifier/** → All models & datasets.  
- **training/keypoint_classification_EN.ipynb** → Training notebook.  
- **requirements.txt** → Runtime dependencies.  
- **requirements_training.txt** → Training dependencies.  
- **README.md** → Documentation (this file).

---

## Requirements
- Python 3.9+ recommended  
- Tested on **Windows 11** with RTX 3050 4GB  
- Dependencies:  
  - `opencv-python`  
  - `mediapipe`  
  - `tensorflow`  
  - `numpy`  
  - `pandas` (for training)  
  - `scikit-learn` (for training)  
  - `matplotlib` (optional for training plots)  

---

## Closing the App
- Press **ESC** to close the window.  
- If running in terminal → `Ctrl + C`.  

---

## Future Improvements
- Support for dynamic gestures (sequences).
- Multi-hand gesture recognition.
- Better visualization for gesture confidence.

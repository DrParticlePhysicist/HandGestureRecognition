# ✋ Real-Time Hand Gesture Recognition

**Author:** Ritik Awasthi 

This project implements **real-time static hand gesture recognition** using **MediaPipe, OpenCV, and TensorFlow Lite**.  
Demo Video: [Click Here](https://drive.google.com/file/d/1BZ6viH2rCFhZeYTYkAvUNLq7iehFB6VT/view?usp=drivesdk)

---

## ✨ Features
- Detects a single hand in real time using **MediaPipe Hands**.
- Classifies gestures using a trained **TensorFlow Lite model**.
- Supports **logging mode** to collect custom training data (`keypoint.csv`).
- Custom gestures can be added by collecting data and retraining the model.
- Toggle between **inference mode** and **logging mode** with a single key.
- Lightweight and runs in real-time even on a laptop GPU/CPU.

---

## 👨‍💻 Technology Justification
- **OpenCV:** Used for real-time video capture and visualization. It provides reliable camera access, frame handling, and efficient image processing.  
- **MediaPipe Hands:** Chosen for robust hand detection and extraction of 21 landmarks. It is lightweight, highly accurate, and optimized for real-time CPU/GPU inference, eliminating the need for a custom detection pipeline.  
- **TensorFlow / TensorFlow Lite:** Used to train and deploy a neural network classifier on the preprocessed landmarks. TensorFlow Lite makes the trained model efficient for real-time inference and portable across devices.  

👉 This stack provides the **best trade-off between speed, accuracy, and ease of integration**, which is crucial for real-time gesture recognition.

---

## ✋ Gesture Logic Explanation
The system identifies gestures by analyzing normalized 21-point hand landmarks from MediaPipe. These landmarks form a structured vector (42 dimensions), which is classified by a neural network.  

- **Fist (✊):** All fingers folded, fingertips positioned close to the palm region.  
- **Open Palm (🖐):** All five fingers extended and spread out.  
- **Peace (✌):** Index and middle fingers extended, ring and little fingers folded.  
- **Thumbs Up (👍):** Thumb extended upward, other fingers folded toward the palm.  

Each pattern is logged as numerical landmark data (`keypoint.csv`) during dataset creation, then used to train the classifier.  

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/DrParticlePhysicist/HandGestureRecognition.git
cd HandGestureRecognition
```

### 2. Create virtual environment
```bash
python -m venv .venv
.\.venv\Scriptsctivate   # (Windows)
source .venv/bin/activate  # (Linux/Mac)
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

## ▶️ Running the Application
```bash
python app.py
```

### Controls:
- **Default = Inference Mode** (if a trained model exists).  
- **Press `k`** → Toggle Logging mode ON/OFF.  
- **Press `0–9` keys** → Assign current gesture to that number label (saved in `keypoint.csv`).  
- **Press `ESC`** → Exit the program.  
- **Ctrl + C** if running in terminal.  

---

## 📚 Training New Gestures
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
5. Restart `app.py` → app now classifies your gestures in real time 🎉

---

## 📂 File Overview
- **app.py** → Main app (inference + logging).  
- **utils.py** → FPS calculator.  
- **model/keypoint_classifier/** → Models & datasets.  
- **training/keypoint_classification_EN.ipynb** → Training notebook.  
- **requirements.txt** → Runtime dependencies.  
- **requirements_training.txt** → Training dependencies.  
- **README.md** → Documentation (this file).  

---

## 🛠 Requirements
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

## ❌ Closing the App
- Press **ESC** in the app window.  
- Or **Ctrl + C** in the terminal.  

---

## 🎥 Demonstration
A short demo video of the application identifying gestures is available here:  
[Watch Demo](https://drive.google.com/file/d/1BZ6viH2rCFhZeYTYkAvUNLq7iehFB6VT/view?usp=drivesdk)

---

## 🚀 Future Improvements
- Extend support for **dynamic gestures** (time-series based).  
- Enable **multi-hand gesture recognition**.  
- Improve visualization with **confidence scores**.

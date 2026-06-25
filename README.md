# 😊 Real-Time Facial Emotion Recognition

A deep learning-based web application that detects and classifies human facial emotions in real time using a webcam or uploaded images — built with TensorFlow and Gradio.

---

## 📌 Overview

This project uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to recognize **7 human emotions** from facial expressions. The application supports two modes:

- **Live Webcam** — Real-time emotion detection via browser camera
- **Image Upload** — Emotion analysis on a static image

---

## 🎭 Emotions Detected

| Label | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

## 🗂️ Project Structure

```
real_time_emotion-recognition/
│
├── app.py                               # Main Gradio application
├── train.ipynb                          # Model training notebook (Google Colab)
├── fer_model.h5                         # Trained CNN model (HDF5 format)
├── haarcascade_frontalface_default.xml  # Face detection classifier
└── requirements.txt                     # Python dependencies
```

---

## 🔗 Model Training — Google Colab

The model was trained on **Google Colab** using a free GPU (T4) runtime for faster computation.

> 📓 Open the training notebook: [`train.ipynb`](./train.ipynb)

The notebook covers the full training pipeline:

- Loading and preprocessing the **FER-2013** dataset
- Data augmentation to improve generalization
- Building and compiling the CNN architecture
- Training with validation and early stopping
- Saving the final model as `fer_model.h5`

To run it yourself, upload `train.ipynb` to [Google Colab](https://colab.research.google.com), enable a GPU runtime (**Runtime → Change runtime type → T4 GPU**), and execute all cells.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Harshu1306/real-time-emotion-recognition.git
cd real-time-emotion-recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```bash
python app.py
```

Then open your browser and go to: `http://127.0.0.1:7860`

---

## 📦 Requirements

```
gradio
tensorflow==2.15.0
numpy
pandas
opencv-python-headless
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Framework | TensorFlow / Keras |
| Architecture | CNN |
| Input Shape | 64 × 64 × 1 (Grayscale) |
| Output Classes | 7 |
| Dataset | FER-2013 |
| Model File | `fer_model.h5` |

Face detection is handled by OpenCV's **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`).

---

## 🖥️ How It Works

1. The webcam feed or uploaded image is converted to grayscale.
2. The Haar Cascade detects face regions in the frame.
3. Each detected face is resized to **64×64** and normalized.
4. The CNN model predicts emotion probabilities across 7 classes.
5. The predicted emotion and confidence score are overlaid on the frame.

---

## 📸 Application Modes

### Live Webcam
- Click **Start** in the app and allow camera access.
- Detected faces are highlighted with a bounding box and emotion label.
- Displays **"No face detected"** if no face is found in the frame.

### Upload Image
- Upload a `.jpg`, `.jpeg`, or `.png` image.
- The app detects all faces and displays emotion predictions with a probability table.

---

## 📊 Model Accuracy

| Metric | Value |
| Test Accuracy | 64.68% |
| Dataset | FER-2013 (35,887 images) |

> The FER-2013 dataset is inherently challenging — even state-of-the-art models typically achieve 65–75% on it due to noisy labels and high visual similarity between certain emotion classes.

---

## ⚠️ Challenges & Limitations

| # | Challenge | Status |
|---|-----------|--------|
| 1 | **Imbalanced Dataset** — Happy and Neutral classes dominate over Fear and Disgust | ⚠️ Dataset limitation; model performs weaker on minority classes |
| 2 | **Real-Time Lag** — Webcam inference is slow on CPU | ⚠️ Partially mitigated; performance depends on hardware |
| 3 | **TensorFlow Version Mismatch** — Model trained on TF 2.20.0 but cloud platforms support older versions | ✅ Re-saved model in `.h5` format compatible with TF 2.15.0 |
| 4 | **Deployment Dependency Conflicts** — Streamlit Cloud failed due to `rich`/`pygments` conflicts | ✅ Migrated from Streamlit to **Gradio** with a simplified stack |
| 5 | **Face Detection Sensitivity** — Strict Haar Cascade settings caused webcam to miss faces | ✅ Tuned `scaleFactor` and `minNeighbors` for better detection |

---

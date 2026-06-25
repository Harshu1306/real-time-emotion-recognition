# 😊 Real-Time Facial Emotion Recognition

A deep learning-based web application that detects and classifies human facial emotions in real time using a webcam or uploaded images — built with TensorFlow and Streamlit.

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
├── app.py                            # Main Streamlit application
├── train.ipynb                       # Model training notebook
├── fer_model.keras                   # Trained CNN model
├── haarcascade_frontalface_default.xml  # Face detection classifier
└── requirements.txt                  # Python dependencies
```

---

## 🔗 Model Training — Google Colab

The model was trained on **Google Colab** using a free GPU runtime for faster computation.

> 📓 Open the training notebook: [`train.ipynb`](./train.ipynb)

The notebook covers the full training pipeline:

- Loading and preprocessing the **FER-2013** dataset
- Data augmentation to improve generalization
- Building and compiling the CNN architecture
- Training with validation and early stopping
- Saving the final model as `fer_model.keras`

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
streamlit run app.py
```

Then open your browser and go to: `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
tensorflow
numpy
pandas
opencv-python
av
streamlit-webrtc
```

> Generate `requirements.txt` with: `pip freeze > requirements.txt`

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Framework | TensorFlow / Keras |
| Architecture | CNN |
| Input Shape | 64 × 64 × 1 (Grayscale) |
| Output Classes | 7 |
| Dataset | FER-2013 |
| Model File | `fer_model.keras` |

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
- Click **START** in the app and allow camera access.
- Detected faces are highlighted with a bounding box and emotion label.

### Upload Image
- Upload a `.jpg`, `.jpeg`, or `.png` image.
- The app detects all faces and displays emotion predictions with a probability bar chart.

---

## 📊 Model Accuracy

| Metric | Value |
| Accuracy | 64.68% |
| Dataset | FER-2013 (35,887 images) |

> The FER-2013 dataset is inherently challenging — even state-of-the-art models typically achieve 65–75% on it due to noisy labels and high visual similarity between certain emotion classes.

---

## ⚠️ Challenges & Limitations

### 1. Imbalanced Dataset
The FER-2013 dataset has a significant class imbalance — emotions like **Happy** and **Neutral** are heavily overrepresented compared to **Disgust** and **Fear**. This causes the model to perform better on frequent classes and struggle with rare ones.

### 2. Low Accuracy on Similar Emotions
Emotions such as **Fear** and **Disgust** share subtle facial muscle patterns, making them visually similar. The model frequently confuses these two classes, which is a known limitation of CNN-based approaches on FER-2013.

### 3. Real-Time Lag / Webcam Issues
Live webcam inference can experience latency depending on the host machine's hardware. The `streamlit-webrtc` library adds browser-level overhead, and inference speed is impacted when running on CPU without GPU acceleration.

---


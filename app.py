import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model("fer_model.keras")

# =====================================
# EMOTIONS
# =====================================

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# =====================================
# FACE DETECTOR
# =====================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_emotion(face):
    face = cv2.resize(face, (64, 64))
    face = face.astype("float32") / 255.0
    face = face.reshape(1, 64, 64, 1)
    preds = model.predict(face, verbose=0)[0]
    idx = np.argmax(preds)
    return EMOTIONS[idx], preds[idx] * 100, preds

# =====================================
# IMAGE PREDICTION
# =====================================

def predict_image(image):
    if image is None:
        return None, "No image provided.", None

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return image, "⚠️ No face detected. Please try another image.", None

    emotion_label = ""
    all_preds = None

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        emotion, confidence, preds = predict_emotion(face)
        all_preds = preds
        emotion_label = f"😊 Detected Emotion: **{emotion}** ({confidence:.1f}%)"

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{emotion} ({confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Build probability dataframe
    prob_df = pd.DataFrame({
        "Emotion": EMOTIONS,
        "Probability (%)": (all_preds * 100).round(2)
    }).set_index("Emotion")

    return image, emotion_label, prob_df

# =====================================
# WEBCAM PREDICTION
# =====================================

def predict_webcam(frame):
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            emotion, confidence, _ = predict_emotion(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emotion} ({confidence:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        except Exception:
            pass

    return frame

# =====================================
# GRADIO UI
# =====================================

with gr.Blocks(title="😊 Facial Emotion Recognition") as demo:

    gr.Markdown(
        """
        # 😊 Facial Emotion Recognition System
        Detects emotions from facial expressions using a CNN trained on the FER-2013 dataset.
        Choose a mode below — **Upload Image** or **Live Webcam**.
        """
    )

    with gr.Tabs():

        # ---------------------------------
        # TAB 1 — Upload Image
        # ---------------------------------
        with gr.TabItem("📸 Upload Image"):

            gr.Markdown("Upload a face image to detect the emotion.")

            with gr.Row():
                input_image = gr.Image(
                    type="numpy",
                    label="Input Image"
                )
                output_image = gr.Image(
                    type="numpy",
                    label="Prediction Result"
                )

            emotion_text = gr.Markdown(label="Detected Emotion")
            prob_table = gr.Dataframe(
                label="📊 Emotion Probabilities (%)",
                headers=["Probability (%)"]
            )

            predict_btn = gr.Button("🔍 Predict Emotion", variant="primary")

            predict_btn.click(
                fn=predict_image,
                inputs=input_image,
                outputs=[output_image, emotion_text, prob_table]
            )

        # ---------------------------------
        # TAB 2 — Live Webcam
        # ---------------------------------
        with gr.TabItem("🎥 Live Webcam"):

            gr.Markdown("Allow camera access and click **Start** for real-time emotion detection.")

            gr.Interface(
                fn=predict_webcam,
                inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Feed"),
                outputs=gr.Image(type="numpy", label="Detected Emotion"),
                live=True
            )

    gr.Markdown("---")
    gr.Markdown("Built with TensorFlow, OpenCV, and Gradio.")

# =====================================
# LAUNCH
# =====================================

demo.launch()
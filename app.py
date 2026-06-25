import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model("fer_model.h5")

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

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # less strict
        minNeighbors=3,    # less strict
        minSize=(30, 30)
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
        return frame

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # less strict — detects more faces
        minNeighbors=3,    # less strict
        minSize=(30, 30)
    )

    if len(faces) == 0:
        cv2.putText(
            output,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        return output

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            emotion, confidence, _ = predict_emotion(face)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                output,
                f"{emotion} ({confidence:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        except Exception:
            pass

    return output

# =====================================
# GRADIO UI
# =====================================


with gr.Blocks(title="😊 Facial Emotion Recognition") as demo:

    gr.Markdown(
        """
        # 😊 Facial Emotion Recognition System
        
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

            gr.Markdown("Allow camera access for real-time emotion detection.")

            gr.Interface(
                fn=predict_webcam,
                inputs=gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam Feed"
                ),
                outputs=gr.Image(
                    type="numpy",
                    label="Detected Emotion"
                ),
                live=True
            )

    gr.Markdown("---")


# =====================================
# LAUNCH
# =====================================

demo.launch()

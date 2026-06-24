import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Facial Emotion Recognition System")

# =====================================
# LOAD MODEL
# =====================================


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fer_model.keras")


model = load_model()

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

    face = face.reshape(
        1,
        64,
        64,
        1
    )

    preds = model.predict(
        face,
        verbose=0
    )[0]

    idx = np.argmax(preds)

    emotion = EMOTIONS[idx]

    confidence = preds[idx] * 100

    return emotion, confidence, preds

# =====================================
# SIDEBAR
# =====================================


mode = st.sidebar.radio(
    "Choose Mode",
    [
        "Live Webcam",
        "Upload Image"
    ]
)

# =====================================
# VIDEO PROCESSOR
# =====================================


class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = gray[
                y:y+h,
                x:x+w
            ]

            try:

                emotion, confidence, _ = predict_emotion(
                    face
                )

                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{emotion} "
                    f"({confidence:.1f}%)"
                )

                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            except Exception:
                pass

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =====================================
# LIVE WEBCAM MODE
# =====================================


if mode == "Live Webcam":

    st.subheader(
        "🎥 Real-Time Emotion Detection"
    )

    st.info(
        "Click START and allow "
        "camera permissions."
    )

    webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

# =====================================
# IMAGE UPLOAD MODE
# =====================================

elif mode == "Upload Image":

    st.subheader(
        "📸 Upload Face Image"
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:

            st.warning(
                "No face detected."
            )

        else:

            all_preds = None

            for (x, y, w, h) in faces:

                face = gray[
                    y:y+h,
                    x:x+w
                ]

                emotion, confidence, preds = (
                    predict_emotion(face)
                )

                all_preds = preds

                cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"{emotion} "
                    f"({confidence:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Prediction Result",
                use_container_width=True
            )

            st.success(
                f"Detected Emotion: "
                f"{emotion} "
                f"({confidence:.1f}%)"
            )

            st.subheader(
                "📊 Emotion Probabilities"
            )

            prob_df = pd.DataFrame(
                {
                    "Emotion": EMOTIONS,
                    "Probability (%)":
                    all_preds * 100
                }
            )

            st.bar_chart(
                prob_df.set_index(
                    "Emotion"
                )
            )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

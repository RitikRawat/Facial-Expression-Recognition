import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model (assumes you have a trained Keras model saved as .h5)
@st.cache_resource

def load_model():
    return tf.keras.models.load_model("facial_expression_model.h5")

model = load_model()

# Define label map (FER2013 emotions)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None, frame

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        return face, (x, y, w, h)

    return None, frame

st.title("Facial Expression Recognition")
st.write("This app uses your webcam to detect facial expressions in real time.")

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    face_input, box = preprocess_frame(frame)
    
    if face_input is not None:
        prediction = model.predict(face_input)
        emotion = emotion_labels[np.argmax(prediction)]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Webcam is stopped')
    camera.release()

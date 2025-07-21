import streamlit as st
import os
import gdown
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "deepfake_model.h5"
FILE_ID = "1iMF-A015RHMZQNE2EYBG9U1sRMINB1YU"  # ← Replace with your real Google Drive file ID

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = download_and_load_model()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32")
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def is_deepfake(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]
    if prediction > 0.5:
        return "Fake", prediction
    else:
        return "Real", 1 - prediction

# UI
st.title("🕵️ Foolproof Deepfake Detector")
st.write("Upload an image to check if it's real or fake using our AI model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.image(temp_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        try:
            result, confidence = is_deepfake(temp_path)
            st.success(f"Result: **{result}**\n\nConfidence: **{confidence:.2f}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    os.remove(temp_path)

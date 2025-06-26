import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Title
st.title("🕵️ Deepfake Detector")
st.write("Upload an image to check if it's real or fake.")

# Load model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepfake_model.h5")
    return model

model = load_model()
threshold = 0.53  # Hardcoded threshold

# Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
     if image.size[0] > 2000 or image.size[1] > 2000:
        image = image.resize((1024, 1024))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction)

    # Show result
    st.markdown(f"**Detection Confidence:** {confidence * 100:.2f}%")

    if confidence >= threshold:
        st.markdown("### 🔥 **Fake Image Detected**")
    else:
        st.markdown("### ✅ **Real Image Detected**")

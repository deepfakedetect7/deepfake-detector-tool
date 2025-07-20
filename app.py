import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Title
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️")
st.title("🕵️ Foolproof Deepfake Detector")
st.write("Upload an image to check if it's **real** or **fake** using our AI model.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_model.h5")

model = load_model()

# Optional: Let users tweak threshold
threshold = st.slider("Detection Threshold (Default = 50%)", 0.0, 1.0, 0.5)

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        original_size = image.size
        st.image(image, caption=f"Uploaded Image ({original_size[0]}x{original_size[1]})", use_column_width=True)

        # Resize for model (without destroying aspect ratio)
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction)

        # Result display
        st.markdown(f"### 🔍 Detection Confidence: **{confidence * 100:.2f}%**")

        if confidence >= threshold:
            st.markdown("### 🔥 **Result: Fake Image Detected**")
        else:
            st.markdown("### ✅ **Result: Real Image Detected**")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

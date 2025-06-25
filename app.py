import streamlit as st
from deepfake_detector import is_deepfake
from PIL import Image
import io

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read file as bytes
    image_bytes = uploaded_file.read()

    # Display image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save bytes to disk for OpenCV to read
    with open("temp.jpg", "wb") as f:
        f.write(image_bytes)

    # Run prediction
    result, confidence = is_deepfake("temp.jpg")
    st.write(f"### Result: {result}")
    st.write(f"### Confidence: {confidence:.2f}")

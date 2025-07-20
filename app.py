import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Configure page
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️")
st.title("🕵️ Foolproof Deepfake Detector")
st.write("Upload an image to check if it's **real** or **fake** using our AI model.")

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_model.h5")

model = load_model()
threshold = 0.51  # Fixed threshold

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load and show original image
        image = Image.open(uploaded_file).convert("RGB")
        original_size = image.size
        st.image(image, caption=f"Uploaded Image ({original_size[0]}x{original_size[1]})", use_container_width=True)

        # Preprocess for model input
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction)

        # Show result
        st.markdown(f"### 🔍 Detection Confidence: **{confidence * 100:.2f}%**")
        if confidence >= threshold:
            st.markdown("### 🔥 **Result: Fake Image Detected**")
        else:
            st.markdown("### ✅ **Result: Real Image Detected**")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the image: {e}")

# Tips for best results
st.markdown("""
<details>
<summary>💡 Tips for Best Results</summary>

- Upload clear, front-facing facial images.
- Avoid low-resolution, blurry, or overly compressed images.
- This is a public-use tool and not suitable for legal or forensic verification.

</details>
""", unsafe_allow_html=True)

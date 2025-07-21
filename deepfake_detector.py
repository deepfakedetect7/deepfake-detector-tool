import os
import cv2
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# 🔁 Automatically download model if not already present
model_path = "deepfake_model.h5"
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    file_id = "1iMF-A015RHMZQNE2EYBG9U1sRMINB1YU"  # 🔁 Replace this with your actual file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
else:
    print("Model already exists. Skipping download.")

# 🔍 Load the trained model
model = load_model(model_path)

# 🧼 Image preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (224, 224))           # Resize to 224x224
    image = image.astype("float32")
    image = preprocess_input(image)                 # EfficientNet preprocessing
    image = np.expand_dims(image, axis=0)
    return image

# 🤖 Predict real or fake
def is_deepfake(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        return "Fake", confidence
    else:
        return "Real", 1 - confidence

# 🧪 Example usage
if __name__ == "__main__":
    test_image = "test_image.jpg"  # 🔁 Replace with your test image path
    try:
        result, confidence = is_deepfake(test_image)
        print(f"Result: {result}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error: {e}")

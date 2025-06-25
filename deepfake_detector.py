import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model (placeholder)
model = load_model("deepfake_model.h5")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def is_deepfake(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        return "Fake", prediction[0][0]
    else:
        return "Real", 1 - prediction[0][0]

# Example usage
if __name__ == "__main__":
    result, confidence = is_deepfake("test_image.jpg")
    print(f"Result: {result}, Confidence: {confidence:.2f}")

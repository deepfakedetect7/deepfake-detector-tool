import cv2
from deepfake_detector import is_deepfake

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            temp_path = "frame.jpg"
            cv2.imwrite(temp_path, frame)
            result, confidence = is_deepfake(temp_path)
            print(f"Frame {frame_count}: {result}, Confidence: {confidence:.2f}")
            if result == "Fake":
                fake_count += 1

    cap.release()
    if fake_count > 0:
        print("Deepfake detected.")
    else:
        print("Video is likely real.")

# Example usage
if __name__ == "__main__":
    process_video("sample_video.mp4")

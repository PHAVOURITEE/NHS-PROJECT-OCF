import cv2
import os
import numpy as np

# Load the Haarcascade classifier for license plate detection
CASCADE_PATH = r'C:\Users\User\NHS PROJECT OCF\nhs-project\haarcascade_licence_plate_rus_16stages.xml'
plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Set up video capture
VIDEO_PATH = r'C:\Users\User\NHS PROJECT OCF\nhs-project\motor.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

# Create a folder to save snapshots
OUTPUT_FOLDER = 'snapshots'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_license_plates(frame):
    """Detects license plates in the given frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    return plates

def simple_hash(image):
    """Computes a simple hash of the given image."""
    return hash(image.tobytes())

# Initialize variables
frame_count = 0
plates_count = 0
saved_hashes = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # Detect license plates in the frame
    plates = detect_license_plates(frame)

    if len(plates) > 0:
        plates_count += 1
        print(f"Frame {frame_count}: {len(plates)} license plate(s) detected")

        for i, (x, y, w, h) in enumerate(plates):
            plate_img = frame[y:y+h, x:x+w]
            plate_hash = simple_hash(plate_img)

            if plate_hash not in saved_hashes:
                snapshot_path = os.path.join(OUTPUT_FOLDER, f'plate_{plates_count}_{i}.jpg')
                cv2.imwrite(snapshot_path, plate_img)
                print(f"Saved snapshot: {snapshot_path}")
                saved_hashes.add(plate_hash)

    frame_count += 1

    # Add a small delay to control processing speed
    if frame_count % 30 == 0:  # Every 30 frames
        print(f"Processed {frame_count} frames. Press Enter to continue or 'q' to quit...")
        if input().strip().lower() == 'q':
            break

# Release the video capture
cap.release()

print(f"Video processing completed. Processed {frame_count} frames, detected license plates in {plates_count} frames.")

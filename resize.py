import cv2
from ultralytics import YOLO
import numpy as np
import pyautogui
import time

# Load your YOLOv8 model
model = YOLO("/home/vmukti/Desktop/Helmet_detect/runs/detect/H13/weights/best.pt")

# Define class names (ensure they match the order in your data.yaml file)
class_names = ['With Helmet']

# Confidence threshold
confidence_threshold = 0.8

# Open the webcam
cap = cv2.VideoCapture('rtsp://admin:@192.168.1.8:554/ch0_0.264')

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the screen size for video writing
screen_size = (640, 480)  # Adjust according to your requirements
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for saving the video
out = cv2.VideoWriter("screen_recording.avi", fourcc, 20.0, screen_size)

# Give some time before starting the recording
time.sleep(2)

print("Recording... Press 'q' to stop.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break
        
    # Resize the frame to 640x480 (if needed)
    frame_resized = cv2.resize(frame, screen_size)

    # Perform object detection
    results = model(frame_resized)

    # Iterate over each detection and draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # get bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # get confidences
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # get class ids

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            # Check if confidence is greater than the threshold
            if confidence > confidence_threshold and class_id < len(class_names):
                # Get the class name
                class_name = class_names[class_id]

                # Draw the bounding box
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_name}: {confidence:.2f}"

                # Set color based on class name
                color = (0, 255, 0) if class_name == "With Helmet" else (0, 0, 255)  # Green for "With Helmet", Red for "Without Helmet"
                
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Write the frame to the video file
    out.write(frame_resized)

    # Display the frame with detections
    cv2.imshow("YOLOv8 Object Detection", frame_resized)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Recording stopped and saved as 'screen_recording.avi'.")

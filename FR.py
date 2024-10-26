import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO("/home/vmukti/Desktop/Helmet_detect/best.pt")

# Confidence threshold
confidence_threshold = 0.7

# Open the webcam or use CCTV link (change to your preferred input)
use_webcam = True  # Set to False if using a CCTV link
if use_webcam:
    cap = cv2.VideoCapture('rtsp://admin:@RTSP-ATPL-900258-AIPTZ.torqueverse.dev:258/ch0_0.264')  # Use 0 for default webcam
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform object detection
    results = model(frame)

    # Iterate over each detection and draw bounding boxes for those above confidence threshold
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Get class IDs
        class_names = result.names  # Get the label names (names are already in your model)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            # Only show results with confidence greater than the threshold
            if confidence > confidence_threshold:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                label = class_names[class_id]  # Use the label (name) of the detected class
                color = (0, 255, 0)  # Green for detection

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame with detections
    cv2.imshow("YOLOv8 Object Detection with Names", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video feed and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
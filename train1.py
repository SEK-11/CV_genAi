from ultralytics import YOLO

# Load the YOLOv8 model (you can start with a pre-trained model like YOLOv8n)
model = YOLO('yolov8l.pt')  # Or choose a different variant

# Train the model
model.train(
    data='/home/vmukti/Desktop/Helmet_detect/dataset_yolo/data.yaml',  # Path to your data.yaml file
    epochs=100,  # Number of epochs
    imgsz=640,  # Image size (e.g., 640x640)
    batch=16,  # Batch size
    name='H1'  # Experiment name
)
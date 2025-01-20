from ultralytics import YOLO

# Load a YOLOv8 model (pretrained weights or a new model)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for the Nano model or other sizes like 'yolov8s.pt'

# Train the model
model.train(
    data='/net/travail/rramesh/TRDP v2/custom.yaml',  # Path to the YAML file
    epochs=150,         # Number of epochs to train
    imgsz=640,         # Image size for training
    batch=16,          # Batch size
    workers=4,         # Number of dataloader workers
    device=0           # GPU (use 'cpu' for CPU training)
)
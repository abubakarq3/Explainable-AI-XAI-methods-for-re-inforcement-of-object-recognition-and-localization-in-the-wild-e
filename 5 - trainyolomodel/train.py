import argparse
from ultralytics import YOLO

def train_yolov8(model_path, data_path, epochs, img_size, batch_size, workers, device):
    """
    Train a YOLOv8 model with specified parameters.

    Args:
        model_path (str): Path to the YOLO model weights file.
        data_path (str): Path to the YAML file specifying training data.
        epochs (int): Number of epochs to train.
        img_size (int): Image size for training.
        batch_size (int): Batch size for training.
        workers (int): Number of dataloader workers.
        device (str): Device to use ('0' for GPU, 'cpu' for CPU).
    """
    model = YOLO(model_path)
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        workers=workers,
        device=device
    )

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model weights file (e.g., 'yolov8n.pt').")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the YAML file specifying training data.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train (default: 150).")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for training (default: 640).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16).")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers (default: 4).")
    parser.add_argument("--device", type=str, default="0", help="Device to use ('0' for GPU, 'cpu' for CPU).")
    args = parser.parse_args()

    train_yolov8(
        model_path=args.model_path,
        data_path=args.data_path,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device
    )

if __name__ == "__main__":
    main()
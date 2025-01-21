from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Use a pretrained YOLO model (nano version for speed)

model.train(
    data='C:/Users/unduood/Desktop/starwarsproj/starwars.yaml',  # Path to dataset YAML file
    epochs=50,                                                # Start with 50 epochs
    imgsz=640,                                                # Resize all images to 640x640
    batch=4,                                                  # Reduce batch size due to small dataset
    rect=True,                                                # Preserve aspect ratio with padding
    augment=True,                                             # Use data augmentation to improve generalization
    #workers=4                                                 # Use multiple workers to speed up data loading (adjust if needed)
)
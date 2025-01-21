'''
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Use a pretrained YOLO model (nano version for speed)

# Train the model
model.train(data='starwars.yaml', epochs=50, imgsz=640, batch=8)
'''

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='C:/Users/unduood/Desktop/starwarsproj/starwars.yaml', epochs=50, imgsz=640, batch=8)


import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO('runs/detect/train4/weights/best.pt')  # Path to the best weights from training

# Load and predict on an unseen image
image_path = 'unseen_images/sw5.jpg'
results = model(image_path, conf=0.1)  # Lower confidence threshold for debugging

'''
# Check if any boxes are detected (for debugging)
if results[0].boxes:
  print("Detected boxes:", results[0].boxes)  # Print detected boxes and details
'''

# Extract the annotated image directly from the results (this adds bounding boxes and labels)
annotated_image = results[0].plot()  # `plot()` method should annotate the image with boxes and labels

# Convert the image to RGB for correct display (OpenCV uses BGR, while Matplotlib uses RGB)
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes and labels
plt.imshow(annotated_image)
plt.axis('off')  # Turn off axis
plt.title("Detected Characters")
plt.show()
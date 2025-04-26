# Install libraries first
# pip install ultralytics opencv-python matplotlib

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the model
model = YOLO('yolo11m-seg.pt')  # Make sure you have the model!

# Load an image
image_path = '../assets/hand.png'  # Change this
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Predict
results = model.predict(img_rgb, imgsz=640, conf=0.5)

# Visualize
res_plotted = results[0].plot()

plt.figure(figsize=(12, 8))
plt.imshow(res_plotted)
plt.axis('off')
plt.title('YOLO11m-seg Segmentation Output')
plt.show()

from ultralytics import YOLO
import cv2

# Load YOLOv11m-seg model (make sure you have the .pt file in the working directory)
model = YOLO('yolo11m-seg.pt')  # Replace with the correct path if needed

# Start the webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Loop to read frames and run segmentation
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run segmentation on the frame
    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

    # Plot results on the frame (segmentation masks)
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv11m-seg Webcam", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
